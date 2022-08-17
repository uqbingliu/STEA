# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
#from emea.one2one_checker_v2 import One2OneCheckerV2
from stea.conf import Config
from stea.data import load_data_for_kg, load_alignment
from stea.data import KG
import os
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, Dataset
from stea.emea_framework import JointDistrModule
from stea.indicator_paris import ParisIndicator as ParisModel
from stea.indicator_paris import Data as ParisData


class EMAlignLabelDataset(Dataset):
    def __init__(self, alignment):
        self.alignment = alignment

    def __len__(self):
        return len(self.alignment)

    def __getitem__(self, item):
        pair = self.alignment[item]
        return torch.tensor(pair[0]), torch.tensor(pair[1])


class Hybrid3JointDistri(nn.Module, JointDistrModule):
    def __init__(self, conf: Config):
        super(Hybrid3JointDistri, self).__init__()
        self.conf = conf
        kgid1, kgid2 = self.conf.data_name.split("_")

        self.kg1 = KG(conf.data_dir, kgid1, add_inverse_edge=True)
        self.kg2 = KG(conf.data_dir, kgid2, add_inverse_edge=True)
        all_alignment = load_alignment(os.path.join(conf.data_dir, "alignment_of_entity.txt"))
        train_alignment = load_alignment(os.path.join(conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(conf.data_dir, "test_alignment.txt"))

        valid_alignment = test_alignment[:100]
        # self.train_alignment = train_alignment + valid_alignment
        self.train_alignment = valid_alignment
        self.labelled_entities = train_alignment + valid_alignment
        self.unlabelled_entities = np.array(test_alignment)[:, 0]

        paris_data = ParisData(self.kg1, self.kg2, all_alignment)
        assert conf.conflict_checker_neigh_num is not None
        self.paris_model = ParisModel(paris_data, conf=conf)
        self.one2one_checker = One2OneCheckerV2(conf)

        rules = self.conf.ruleset.split(",")
        self.feature_num = len(rules)
        self.linear_layer = nn.Linear(in_features=self.feature_num, out_features=1, bias=True)
        # torch.nn.init.ones_(self.linear_layer.weight)
        torch.nn.init.zeros_(self.linear_layer.weight)
        # self.linear_layer.weight = torch.tensor([1, 0])
        torch.nn.init.zeros_(self.linear_layer.bias)

        self.candidates = None
        self.neural_prob_mtx = None

        self.device = torch.device(conf.device)
        self.to(self.device)

    def set_neural_prob_mtx(self, prob_mtx: torch.Tensor):
        self.neural_prob_mtx = prob_mtx
        ent2_num = self.neural_prob_mtx.size()[1]
        for ent1, ent2 in self.labelled_entities:
            tmp_prob = self.neural_prob_mtx[ent1]
            tmp_prob = tmp_prob * 0.5
            tmp_prob[ent2] = tmp_prob[ent2] + 0.5
        self._generate_candidates()
        for ent1, ent2 in self.labelled_entities:
            onehot = torch.zeros(size=(ent2_num,), device=self.device)
            onehot[ent2] = 1
            self.neural_prob_mtx[ent1] = onehot

    def set_stuff_about_neu_model(self, ent1_embs: np.ndarray, prob_mtx: torch.Tensor, all_pred_alignment:list):
        # self.neu_pred_alignment = self.labelled_entities + pred_alignment
        sample_map = dict(all_pred_alignment)
        sample_map.update(dict(self.labelled_entities))
        self.neu_sample_alignment = list(sample_map.items())
        self.neu_all_pred_alignment = all_pred_alignment
        self.set_neural_prob_mtx(prob_mtx)
        self.one2one_checker.set_embs_n_probs(ent1_embs, prob_mtx)
        self.features_cache = None

    def _generate_candidates(self):
        candi_probs, candi_idxes = torch.topk(self.neural_prob_mtx, dim=1, k=self.conf.topK)
        self.candi_sum_prob = torch.sum(candi_probs, dim=1, keepdim=False)
        self.candidates = candi_idxes

    def compute_features(self, sampled_alignment):
        with torch.no_grad():
            ent_arr = torch.arange(0, len(self.candidates), device=self.device)
            fea_list = []

            rule_list = self.conf.ruleset.split(",")

            if "paris" in rule_list:
                features1 = self.paris_model.apply_reasoning_torch(ent_arr, self.candidates, self.neural_prob_mtx)
                features1 = features1.unsqueeze(dim=2)
                fea_list.append(features1)

            # features2 = self.paris_model.apply_negative_reasoning_rule(ent_arr, self.candidates, self.neural_prob_mtx)
            # features = features2.unsqueeze(dim=2)

            if "eqneu" in rule_list or "one2one" in rule_list:
                ent_arr_np = np.arange(0, len(self.candidates))
                features3 = self.one2one_checker.compute_factors_in_batch(ent_arr_np, self.candidates, sampled_alignment, self.neu_all_pred_alignment)
                if "one2one" in rule_list:
                    tmp_fea = features3[:,:,0].unsqueeze(dim=2)
                    fea_list.append(tmp_fea)
                if "eqneu" in rule_list:
                    tmp_fea = features3[:, :, 1].unsqueeze(dim=2)
                    fea_list.append(tmp_fea)
            if len(fea_list) == 0:
                raise Exception("no feature specified")
            elif len(fea_list) == 1:
                features = fea_list[0]
            else:
                features = torch.cat(fea_list, dim=2)
            self.features_cache = features
        return features

    def local_func(self, features: torch.Tensor):
        features = features.to(dtype=torch.float32, device=self.device)
        candi_score_mtx = torch.exp(self.linear_layer(features))
        fea_shape = candi_score_mtx.shape
        candi_score_mtx = candi_score_mtx.reshape(shape=fea_shape[:2])
        return candi_score_mtx

    def conditional_p(self, features: torch.Tensor):
        candi_score_mtx = self.local_func(features)
        factor_prob_mtx = F.normalize(candi_score_mtx, dim=1, p=1)
        return factor_prob_mtx

    # def coordinate_ascend(self):
    #     features = self.compute_features()
    #     candi_prob_mtx = self.conditional_p(features)
    #     hybrid_prob_mtx = self.neural_prob_mtx
    #     hybrid_prob_mtx.scatter(dim=1, index=self.candidates, src=torch.zeros(size=self.candidates.shape, dtype=torch.float32, device=self.device))
    #     hybrid_prob_mtx = F.normalize(hybrid_prob_mtx, dim=1, p=1) * 0.1
    #     candi_prob_mtx = candi_prob_mtx * 0.9
    #     hybrid_prob_mtx.scatter_(dim=1, index=self.candidates, src=candi_prob_mtx)
    #     return hybrid_prob_mtx

    # def coordinate_ascend(self):
    #     prob_q = self.neural_prob_mtx
    #
    #     for ent1 in tqdm(self.unlabelled_entities, desc="coordinate ascending"):
    #         ent1_arr = torch.tensor([ent1], device=self.device)
    #         ent2_candi = self.candidates[ent1].unsqueeze(dim=0)
    #         ent1_features = self.paris_model.apply_reasoning_torch(ent1_arr, ent2_candi, prob_q)
    #         ent1_features = ent1_features.unsqueeze(dim=2)
    #
    #         ent1_q = self.conditional_p(ent1_features)
    #         ent1_q = ent1_q * self.candi_sum_prob[ent1]
    #         prob_q[ent1][ent2_candi] = ent1_q
    #     return prob_q


    def coordinate_ascend(self):
        with torch.no_grad():
            if self.features_cache is None:
                features = self.compute_features(self.neu_pred_alignment)
            else:
                features = self.features_cache
            candi_prob_mtx = self.conditional_p(features)
            hybrid_prob_mtx = self.neural_prob_mtx
            # hybrid_prob_mtx.scatter(dim=1, index=self.candidates, src=torch.zeros(size=self.candidates.shape, dtype=torch.float32, device=self.device))
            # hybrid_prob_mtx = F.normalize(hybrid_prob_mtx, dim=1, p=1) * 0.1
            # candi_prob_mtx = candi_prob_mtx * 0.9
            hybrid_prob_mtx.scatter_(dim=1, index=self.candidates, src=candi_prob_mtx*self.candi_sum_prob.unsqueeze(dim=1))
            np.savez(os.path.join(self.conf.output_dir, "coordinate_ascent_prob_mtx.npz"),  improved_prob_mtx=hybrid_prob_mtx.cpu().numpy())
        return hybrid_prob_mtx

    def train_model(self):
        # opt 1, use expectation of neu prediction
        # sample to generate training data
        # pred_alignment = list()
        # prob_mtx = self.neural_prob_mtx.cpu().numpy()
        # for ent in self.unlabelled_entities:
        #     label = np.random.choice(self.kg2.num_entity, size=1, p=prob_mtx[ent])
        #     pred_alignment.append((ent, label))
        # all_alignment = self.labelled_entities + pred_alignment

        # opt 2, only use labelled data
        # all_alignment = self.labelled_entities
        all_alignment = self.train_alignment
        # opt 3, use neu prediction
        # all_alignment = self.neu_pred_alignment

        candi_arr = self.candidates.cpu().numpy()
        filtered_align_list = []
        for idx in trange(len(all_alignment), desc="filtering alignment"):
            ent1, ent2 = all_alignment[idx]
            candi = candi_arr[ent1]
            ent2_idxes = np.where(candi == ent2)
            if len(ent2_idxes[0]):
                filtered_align_list.append((ent1, ent2_idxes[0][0]))
        train_alignment = np.array(filtered_align_list)
        align_dataset = EMAlignLabelDataset(train_alignment)
        align_dataloader = DataLoader(align_dataset, batch_size=32, shuffle=True)

        # features
        features = self.compute_features(self.neu_sample_alignment)

        # training
        optimizer = torch.optim.Adagrad(params=self.parameters(), lr=1e-2, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

        self.train()
        # while True:
        lowest_loss = None
        no_desc_steps = 0
        for epoch in trange(0, 10000, desc="updating joint distri ..."):
            step_loss_list = []
            for batch in align_dataloader:
                ent1_arr, ent2_arr = batch[0].to(self.device), batch[1].to(self.device)
                sub_features = features[ent1_arr]
                factor_prob_mtx = self.conditional_p(sub_features)
                factor_probs = torch.gather(factor_prob_mtx, dim=1, index=ent2_arr.unsqueeze(dim=1))
                log_factor_probs = torch.log(factor_probs)
                loss = - torch.mean(log_factor_probs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_loss_list.append(loss)

            mean_loss = torch.mean(torch.tensor(step_loss_list))
            scheduler.step(mean_loss)
            mean_loss = mean_loss.cpu().item()
            if lowest_loss is None:
                lowest_loss = mean_loss
            elif mean_loss < lowest_loss:
                lowest_loss = mean_loss
                no_desc_steps = 0
            else:
                no_desc_steps += 1

            if no_desc_steps >=4:
                break
            print(mean_loss, self.linear_layer.weight, self.linear_layer.bias)
        print("finish updating joint distribution")


    # def compute_target_probs(self):
    #     ent_arr = torch.arange(0, len(self.candidates), device=self.device)
    #     candi_mtx = self.candidates
    #     features = self.paris_model.apply_reasoning_torch(ent_arr, candi_mtx, self.neural_prob_mtx)
    #     np.savez(os.path.join(self.conf.output_dir, "paris_prob_mtx.npz"), candi_mtx=candi_mtx.cpu().numpy(), paris_prob_mtx=features.cpu().numpy())
    #     with torch.no_grad():
    #         features = features.unsqueeze(dim=2)
    #         candi_score_mtx = self.forward2(features.to(dtype=torch.float32))
    #     np.savez(os.path.join(self.conf.output_dir, "target_unnorm_prob_mtx.npz"), candi_mtx=candi_mtx.cpu().numpy(), target_prob_mtx=candi_score_mtx.cpu().numpy())
    #     return candi_mtx, candi_score_mtx

    def predict(self):
        with torch.no_grad():
            ent_arr = torch.arange(0, len(self.candidates), device=self.device)
            candi_mtx = self.candidates
            features = self.paris_model.apply_reasoning_torch(ent_arr, candi_mtx, self.neural_prob_mtx)
            features = features.unsqueeze(dim=2)
            candi_score_mtx = self.forward2(features.to(dtype=torch.float32))
        return candi_mtx, candi_score_mtx

    def save(self):
        torch.save(self.state_dict(), os.path.join(self.conf.output_dir, "joint_model.ckpt"))
        self.compute_target_probs()

    def load(self):
        self.load_state_dict(torch.load(os.path.join(self.conf.output_dir, "joint_model.ckpt")))



