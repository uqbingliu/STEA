# -*- coding: utf-8 -*-

from tqdm import tqdm, trange
from stea.data import KG
import numpy as np
from stea.conf import Config
import os
import torch


class Data():
    def __init__(self, kg1: KG, kg2: KG, alignment):
        kg1_triples = kg1.triple_rel_list
        kg2_triples = kg2.triple_rel_list
        # self.kg1_triples = [(h,r,t) for h,t,r in kg1_triples]
        # self.kg2_triples = [(h,r,t) for h,t,r in kg2_triples]
        self.kg1_triples = kg1_triples
        self.kg2_triples = kg2_triples
        self.alignment = alignment
        self.kg1_inbound_map = self.build_inbound_map(self.kg1_triples)
        self.kg2_inbound_map = self.build_inbound_map(self.kg2_triples)
        self.kg1_outbound_map = self.build_outbound_map(self.kg1_triples)
        self.kg2_outbound_map = self.build_outbound_map(self.kg2_triples)

    @staticmethod
    def build_inbound_map(triples):
        inbound_map = dict()
        for h, r, t in triples:
            if t not in inbound_map:
                inbound_map[t] = []
            inbound_map[t].append((r, h))
        return inbound_map

    @staticmethod
    def build_outbound_map(triples):
        outbound_map = dict()
        for h, r, t in triples:
            if h not in outbound_map:
                outbound_map[h] = []
            outbound_map[h].append((r, t))
        return outbound_map


class ParisIndicator:
    def __init__(self, data: Data, conf: Config, inv=False):
        self.conf = conf
        self.data = data
        self.tail_to_relnheads_map1 = data.kg1_inbound_map
        self.tail_to_relnheads_map2 = data.kg2_inbound_map
        self.head_to_relntails_map1 = data.kg1_outbound_map
        self.head_to_relntails_map2 = data.kg2_outbound_map

        paris_cache_arr_fn = os.path.join(conf.data_dir, "paris_cache_arr.npz")

        # if os.path.exists(paris_cache_fn):
        #     with open(paris_cache_fn) as file:
        #         cache_obj = json.loads(file.read())
        #         self.func_map1 = cache_obj["kg1_functionality"]
        #         self.func_map2 = cache_obj["kg2_functionality"]
        #         subrel_map_for_json = cache_obj["subrel"]
        #         self.func_map1 = {int(k): v for k, v in self.func_map1.items()}
        #         self.func_map2 = {int(k): v for k, v in self.func_map2.items()}
        #         self.subrel_map = {eval(k): v for k, v in subrel_map_for_json.items()}

        if os.path.exists(paris_cache_arr_fn):
            print("loading paris cache ...")
            if inv:
                cache_obj_arr = np.load(paris_cache_arr_fn, allow_pickle=True)
                self.func_arr1 = cache_obj_arr["kg2_func_arr"]
                self.func_arr2 = cache_obj_arr["kg1_func_arr"]
                self.sub_rel1_rel2_mtx = cache_obj_arr["sub_rel2_rel1_mtx"]
                self.sub_rel2_rel1_mtx = cache_obj_arr["sub_rel1_rel2_mtx"]
            else:
                cache_obj_arr = np.load(paris_cache_arr_fn, allow_pickle=True)
                self.func_arr1 = cache_obj_arr["kg1_func_arr"]
                self.func_arr2 = cache_obj_arr["kg2_func_arr"]
                self.sub_rel1_rel2_mtx = cache_obj_arr["sub_rel1_rel2_mtx"]
                self.sub_rel2_rel1_mtx = cache_obj_arr["sub_rel2_rel1_mtx"]
            print("complete loading paris cache")
        else:
            self.func_map1, self.func_arr1 = self.functionality(data.kg1_triples)
            self.func_map2, self.func_arr2 = self.functionality(data.kg2_triples)
            self.subrel_map, self.sub_rel1_rel2_mtx, self.sub_rel2_rel1_mtx = self.subsumption(data.kg1_triples, data.kg2_triples, data.alignment)
            # subrel_map_for_json = {str(k): v for k, v in self.subrel_map.items()}
            # cache_obj = {
            #     "kg1_functionality": self.func_map1,
            #     "kg2_functionality": self.func_map2,
            #     "subrel": subrel_map_for_json
            # }
            # with open(paris_cache_fn, "w+") as file:
            #     file.write(json.dumps(cache_obj))
            # del cache_obj
            np.savez(paris_cache_arr_fn, kg1_func_arr=self.func_arr1, kg2_func_arr=self.func_arr2, sub_rel1_rel2_mtx=self.sub_rel1_rel2_mtx, sub_rel2_rel1_mtx=self.sub_rel2_rel1_mtx)
        self.device = torch.device(conf.device)
        self.func_arr1 = torch.tensor(self.func_arr1, device=self.device)
        self.func_arr2 = torch.tensor(self.func_arr2, device=self.device)
        self.sub_rel1_rel2_mtx = torch.tensor(self.sub_rel1_rel2_mtx, device=self.device)
        self.sub_rel2_rel1_mtx = torch.tensor(self.sub_rel2_rel1_mtx, device=self.device)

    def apply_reasoning(self, ent_arr: np.ndarray, candidates_mtx: np.ndarray, prob_mtx: np.ndarray):
        probs_list = []
        row_num, col_num = candidates_mtx.shape
        for row_idx in trange(row_num, desc="run reasoning with paris"):
            ent1 = ent_arr[row_idx]
            probs = []
            for col_idx in range(col_num):
                ent2 = candidates_mtx[row_idx][col_idx]
                relnheads1 = self.tail_to_relnheads_map1[ent1]
                relnheads2 = self.tail_to_relnheads_map2[ent2]
                accu_prob = 1
                for rel1, head1 in relnheads1:
                    for rel2, head2 in relnheads2:
                        rel1_func = self.func_map1[rel1]
                        rel2_func = self.func_map2[rel2]
                        equiv_prob = prob_mtx[head1][head2]
                        sub_prob = (1 - self.subrel_map[(f"kg1:{rel1}", f"kg2:{rel2}")] * rel2_func * equiv_prob) * (
                                1 - self.subrel_map[(f"kg2:{rel2}", f"kg1:{rel1}")] * rel1_func * equiv_prob)
                        accu_prob *= sub_prob
                prob = 1 - accu_prob
                probs.append(prob)
            probs_list.append(probs)
        prob_mtx = np.array(probs_list)
        return prob_mtx

    def apply_reasoning_torch(self, ent_arr: torch.Tensor, candidates_mtx: torch.Tensor, prob_mtx: torch.Tensor):
        ent_arr = ent_arr.to(self.device)
        candidates_mtx = candidates_mtx.to(self.device)
        prob_mtx = prob_mtx  #.to(self.device)
        new_probs_list = []
        row_num, col_num = candidates_mtx.shape
        with torch.no_grad():
            # for row_idx in trange(row_num, desc="run reasoning with paris"):
            for row_idx in trange(row_num, desc="paris reasoning"):
                ent1 = ent_arr[row_idx].cpu().item()
                probs = []
                for col_idx in range(col_num):
                    ent2 = candidates_mtx[row_idx][col_idx].cpu().item()
                    relnheads1 = self.tail_to_relnheads_map1[ent1]
                    relnheads2 = self.tail_to_relnheads_map2[ent2]
                    relnheads1_arr = torch.tensor(relnheads1, device=self.device)
                    relnheads2_arr = torch.tensor(relnheads2, device=self.device)
                    p = self.single_reasoning_torch(relnheads1_arr, relnheads2_arr, prob_mtx)
                    probs.append(p)
                new_probs_list.append(probs)
        new_probs_mtx = torch.tensor(new_probs_list, dtype=torch.float32, device=self.device)
        return new_probs_mtx

    def single_reasoning_torch(self, relnheads1: torch.Tensor, relnheads2: torch.Tensor, prob_mtx: torch.Tensor):
        rel1_arr = relnheads1[:, 0]
        head1_arr = relnheads1[:, 1]
        rel2_arr = relnheads2[:, 0]
        head2_arr = relnheads2[:, 1]

        equiv_prob = prob_mtx[head1_arr.to(prob_mtx.device)][:, head2_arr.to(prob_mtx.device)].to(self.device)
        sub_rel1_rel2 = self.sub_rel1_rel2_mtx[rel1_arr][:, rel2_arr]
        sub_rel2_rel1 = self.sub_rel2_rel1_mtx.transpose(dim0=0, dim1=1)[rel1_arr][:, rel2_arr]
        func1 = self.func_arr1[rel1_arr].unsqueeze(dim=1).repeat(repeats=(1, head2_arr.shape[0]))
        func2 = self.func_arr2[rel2_arr].unsqueeze(dim=0).repeat(repeats=(head1_arr.shape[0], 1))

        sub_probs = (1-sub_rel1_rel2*func2*equiv_prob) * (1-sub_rel2_rel1*func1*equiv_prob)
        prob = 1 - torch.prod(sub_probs)
        return prob

    def apply_negative_reasoning_rule(self, ent_arr: torch.Tensor, candidates_mtx: torch.Tensor, prob_mtx: torch.Tensor):
        ent_arr = ent_arr.to(self.device)
        candidates_mtx = candidates_mtx.to(self.device)
        prob_mtx = prob_mtx.to(self.device)
        new_probs_list = []
        row_num, col_num = candidates_mtx.shape
        with torch.no_grad():
            # for row_idx in trange(row_num, desc="run reasoning with paris"):
            for row_idx in range(row_num):
                ent1 = ent_arr[row_idx].cpu().item()
                probs = []
                for col_idx in range(col_num):
                    ent2 = candidates_mtx[row_idx][col_idx].cpu().item()
                    relntail1 = self.head_to_relntails_map1[ent1]
                    relntail2 = self.head_to_relntails_map2[ent2]
                    relntail1_arr = torch.tensor(relntail1, device=self.device)
                    relntail2_arr = torch.tensor(relntail2, device=self.device)
                    p = self.single_apply_negative_reasoning_rule(ent1, ent2, relntail1_arr, relntail2_arr, prob_mtx)
                    probs.append(p)
                new_probs_list.append(probs)
        new_probs_mtx = torch.tensor(new_probs_list, device=self.device)
        return new_probs_mtx

    def single_apply_negative_reasoning_rule(self, ent1, ent2, relntails1: torch.Tensor, relntails2: torch.Tensor, prob_mtx: torch.Tensor):
        rel1_arr = relntails1[:, 0]
        tail1_arr = relntails1[:, 1]
        rel2_arr = relntails2[:, 0]
        tail2_arr = relntails2[:, 1]

        equiv_prob = prob_mtx[tail1_arr][:, tail2_arr]
        sub_rel1_rel2 = self.sub_rel1_rel2_mtx[rel1_arr][:, rel2_arr]
        sub_rel2_rel1 = self.sub_rel2_rel1_mtx.transpose(dim0=0, dim1=1)[rel1_arr][:, rel2_arr]
        func1 = self.func_arr1[rel1_arr].unsqueeze(dim=1).repeat(repeats=(1, tail2_arr.shape[0]))
        func2 = self.func_arr2[rel2_arr].unsqueeze(dim=0).repeat(repeats=(tail1_arr.shape[0], 1))
        sub_probs = (1-sub_rel1_rel2*func2*(1-equiv_prob)) * (1-sub_rel2_rel1*func1*(1-equiv_prob))
        prob = prob_mtx[ent1][ent2] * torch.prod(sub_probs)
        return prob

    @staticmethod
    def functionality(triple_list):
        rel_to_head2tails_map = dict()
        for head, rel, tail in tqdm(triple_list):
            if rel not in rel_to_head2tails_map:
                rel_to_head2tails_map[rel] = {head: []}
            elif head not in rel_to_head2tails_map[rel]:
                rel_to_head2tails_map[rel][head] = []
            rel_to_head2tails_map[rel][head].append(tail)

        func_map = {}
        for rel in tqdm(rel_to_head2tails_map.keys(), desc="computing functionality"):
            head2tails_map = rel_to_head2tails_map[rel]
            num_head = len(head2tails_map)
            num_head_tail_pair = 0
            for head in head2tails_map.keys():
                num_head_tail_pair += len(head2tails_map[head])
            func = float(num_head) / float(num_head_tail_pair)
            func_map[rel] = func
        rel_num = len(func_map)
        func_list = [func_map[rel] for rel in range(rel_num)]
        return func_map, func_list

    @staticmethod
    def subsumption(triple1_list, triple2_list, alignment):
        ent2_to_ent1_map = dict()
        for ent1, ent2 in alignment:
            ent2_to_ent1_map[ent2] = ent1

        # assign id to each entity
        ent1_list = []
        rel1_list = []
        for head, rel, tail in triple1_list:
            ent1_list.extend([head, tail])
            rel1_list.append(rel)
        ent1_list = list(set(ent1_list))
        rel1_list = sorted(list(set(rel1_list)))
        ent_to_no_map = dict()
        for idx, ent in enumerate(ent1_list):
            ent_to_no_map[ent] = idx
        ent2_list = []
        rel2_list = []
        for head, rel, tail in triple2_list:
            ent2_list.extend([head, tail])
            rel2_list.append(rel)
        ent2_list = list(set(ent2_list))
        rel2_list = sorted(list(set(rel2_list)))
        ent_no_cursor = len(ent1_list)
        for ent2 in ent2_list:
            if ent2 in ent2_to_ent1_map:
                cor_ent1 = ent2_to_ent1_map[ent2]
                ent_to_no_map[ent2] = ent_to_no_map[cor_ent1]
            else:
                ent_to_no_map[ent2] = ent_no_cursor
                ent_no_cursor += 1

        # count
        rel_to_ent_map1 = dict()
        for head, rel, tail in tqdm(triple1_list):
            if rel not in rel_to_ent_map1:
                rel_to_ent_map1[rel] = []
            rel_to_ent_map1[rel].append((ent_to_no_map[head], ent_to_no_map[tail]))

        rel_to_ent_map2 = dict()
        for head, rel, tail in tqdm(triple2_list):
            if rel not in rel_to_ent_map2:
                rel_to_ent_map2[rel] = []
            rel_to_ent_map2[rel].append((ent_to_no_map[head], ent_to_no_map[tail]))

        subrel_map = dict()
        sub_rel1_rel2_mtx = np.zeros(shape=(len(rel1_list), len(rel2_list)))
        sub_rel2_rel1_mtx = np.zeros(shape=(len(rel2_list), len(rel1_list)))
        for rel1 in tqdm(rel1_list, desc="count subrelation"):
            ent_pair_set1 = set(rel_to_ent_map1[rel1])
            for rel2 in rel2_list:
                ent_pair_set2 = set(rel_to_ent_map2[rel2])
                interset = ent_pair_set1.intersection(ent_pair_set2)
                rel1_rel2 = len(interset) / len(ent_pair_set1)
                rel2_rel1 = len(interset) / len(ent_pair_set2)
                subrel_map[(f"kg1:{rel1}", f"kg2:{rel2}")] = rel1_rel2
                subrel_map[(f"kg2:{rel2}", f"kg1:{rel1}")] = rel2_rel1
                sub_rel1_rel2_mtx[rel1][rel2] = rel1_rel2
                sub_rel2_rel1_mtx[rel2][rel1] = rel2_rel1
        del ent2_to_ent1_map
        del ent_to_no_map
        del rel_to_ent_map1
        del rel_to_ent_map2
        return subrel_map, sub_rel1_rel2_mtx, sub_rel2_rel1_mtx


