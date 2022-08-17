# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
from stea.conf import Config
import os
import abc
from stea.simi_to_prob import SimiToProbModule
import json


class NeuralEAModule:
    def __init__(self, conf: Config):
        self.conf = conf

    @abc.abstractmethod
    def prepare_data(self):
        pass

    @abc.abstractmethod
    def train_model_with_observed_labels(self):
        pass

    @abc.abstractmethod
    def enhance_latent_labels_with_jointdistr(self, improved_candi_probs, candi_mtx):
        pass

    @abc.abstractmethod
    def train_model_with_observed_n_latent_labels(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def get_embeddings(self):
        pass

    @abc.abstractmethod
    def get_pred_alignment(self):
        pass

    def get_all_pred_alignment(self):
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json")) as file:
            obj = json.loads(file.read())
            pred_alignment = obj["pred_alignment_csls"]
            all_pred_alignment = obj["all_pred_alignment_csls"]
        return pred_alignment, all_pred_alignment

    @abc.abstractmethod
    def evaluate(self):
        pass

    def convert_simi_to_probs1(self, simi_mtx: np.ndarray):  # softmax
        device = torch.device(self.conf.device)
        simi_mtx = torch.tensor(simi_mtx, device=device)
        prob_mtx = F.softmax(simi_mtx, dim=1)
        return prob_mtx

    def convert_simi_to_probs2(self, simi_mtx: np.ndarray):  # topk softmax + other zero
        device = torch.device(self.conf.device)
        simi_mtx = torch.tensor(simi_mtx, device=device)
        unnorm_prob_mtx = torch.exp(simi_mtx)
        topk_values, topk_idxes = torch.topk(unnorm_prob_mtx, dim=1, k=self.conf.topK)
        topk_values = F.normalize(topk_values, p=1, dim=1)
        neural_prob_mtx = torch.zeros(size=unnorm_prob_mtx.shape, device=device)
        neural_prob_mtx.scatter_(dim=1, index=topk_idxes, src=topk_values)
        return neural_prob_mtx

    def convert_simi_to_probs3(self, simi_mtx: np.ndarray): # 0.9 * topk softmax + 0.1 * other softmax
        device = torch.device(self.conf.device)
        simi_mtx = torch.tensor(simi_mtx, device=device)
        unnorm_prob_mtx = torch.exp(simi_mtx)
        topk_values, topk_idxes = torch.topk(unnorm_prob_mtx, dim=1, k=self.conf.topK)
        topk_values = F.normalize(topk_values, p=1, dim=1) * 0.9
        unnorm_prob_mtx.scatter_(dim=1, index=topk_idxes, src=torch.tensor(0.0, device=device))
        neural_prob_mtx = F.normalize(unnorm_prob_mtx, dim=1, p=1) * 0.1
        neural_prob_mtx.scatter_(dim=1, index=topk_idxes, src=topk_values)
        return neural_prob_mtx

    def convert_simi_to_probs4(self, simi_mtx: np.ndarray):
        with torch.no_grad():
            device = torch.device(self.conf.device)
            simi_mtx = torch.tensor(simi_mtx, device=device)
            unnorm_prob_mtx = torch.exp(simi_mtx / torch.tensor(self.conf.cross_entropy_tau, device=device))
            topk_values, topk_idxes = torch.topk(unnorm_prob_mtx, dim=1, k=self.conf.topK)
            topk_values = F.normalize(topk_values, p=1, dim=1)
            neural_prob_mtx = torch.zeros(size=simi_mtx.shape, device=device)
            neural_prob_mtx.scatter_(dim=1, index=topk_idxes, src=topk_values)
        return neural_prob_mtx

    def convert_simi_to_probs5(self, simi_mtx: np.ndarray):
        device = torch.device(self.conf.device)
        simi_mtx = torch.tensor(simi_mtx, device=device)
        idxes = simi_mtx.argmax(dim=1, keepdim=True)
        prob_mtx = torch.zeros(size=simi_mtx.shape, device=device)
        prob_mtx.scatter_(dim=1, index=idxes, src=torch.ones(size=idxes.shape, device=device))
        return prob_mtx

    def convert_simi_to_probs6(self, simi_mtx: np.ndarray):
        device = torch.device(self.conf.device)
        simi_mtx = torch.tensor(simi_mtx, device=device)
        prob_mtx = F.softmax(simi_mtx, dim=1)
        idxes = simi_mtx.argmax(dim=1, keepdim=True)
        prob_mtx.scatter_(dim=1, index=idxes, src=torch.zeros(size=idxes.shape, device=device))
        prob_mtx = F.normalize(prob_mtx, p=1, dim=1) * 0.1
        prob_mtx.scatter_(dim=1, index=idxes, src=torch.ones(size=idxes.shape, device=device)*0.9)
        return prob_mtx

    def convert_simi_to_probs7(self, simi_mtx: np.ndarray, inv=False):  # SimiToProbModel
        simi2prob_module = SimiToProbModule(self.conf)
        if inv:
            probs = simi2prob_module.predict(simi_mtx.transpose())
        else:
            probs = simi2prob_module.predict(simi_mtx)
        return probs


class JointDistrModule:
    def __init__(self, conf: Config):
        self.conf = conf

    # @abc.abstractmethod
    # def set_neural_prob_mtx(self, prob_mtx):
    #     pass

    @abc.abstractmethod
    def set_stuff_about_neu_model(self, **kwargs):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def coordinate_ascend(self):
        pass



def compute_metrics(ranking_list, rels):
    gold_rank_list = []
    for idx, rel in enumerate(rels):
        ranking = ranking_list[idx]
        gold_rank = np.where(ranking == rel)[0][0]
        gold_rank_list.append(gold_rank+1)
    gold_rank_arr = np.array(gold_rank_list)
    mean_rank = np.mean(gold_rank_arr)
    mrr = np.mean(1.0/gold_rank_arr)
    recall_1 = np.mean((gold_rank_arr <= 1).astype(np.float32))
    recall_5 = np.mean((gold_rank_arr <= 5).astype(np.float32))
    recall_10 = np.mean((gold_rank_arr <= 10).astype(np.float32))
    recall_50 = np.mean((gold_rank_arr <= 50).astype(np.float32))
    return mean_rank, mrr, recall_1, recall_5, recall_10, recall_50


def evaluate_models(score_mtx: np.ndarray, eval_alignment):
    eval_alignment_arr = np.array(eval_alignment)
    rel_score_mtx = score_mtx[eval_alignment_arr[:, 0]][:, eval_alignment_arr[:, 1]]
    pred_ranking = np.argsort(-rel_score_mtx, axis=1)
    pred_ranking = eval_alignment_arr[:, 1][pred_ranking]
    rels = eval_alignment_arr[:, 1]
    mr, mrr, recall_1, recall_5, recall_10, recall_50 = compute_metrics(pred_ranking, rels)
    print(f"mr:{mr}, mrr:{mrr}, recall@1:{recall_1}, recall@5:{recall_5}, recall@10:{recall_10}, recall@50:{recall_50}")
    metrics = {
        "mr": float(mr),
        "mrr": float(mrr),
        "recall@1": float(recall_1),
        "recall@5": float(recall_5),
        "recall@10": float(recall_10),
        "recall@50": float(recall_50)
    }
    return metrics




