import multiprocessing

import gc
import os

import numpy as np
import time
import torch
from tqdm import trange


class Evaluator():
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.batch_size = 512

    def evaluate_csls(self, embed1: np.ndarray, embed2: np.ndarray, eval_alignment, top_k=(1, 5, 10, 50)):
        t1 = time.time()
        csls_sim_mat = self.csls_sim(embed1, embed2, k=10)
        csls_metrics, pred_alignment = self.compute_metrics(csls_sim_mat, eval_alignment, top_k)
        del csls_sim_mat
        gc.collect()
        t2 = time.time()
        print(f"evaluation speeds: {t2 - t1}s")
        return csls_metrics, pred_alignment

    def evaluate_cosine(self, embed1: np.ndarray, embed2: np.ndarray, eval_alignment, top_k=(1, 5, 10, 50)):
        cos_sim_mat = self.cosine_sim(embed1, embed2)
        cos_metrics, pred_alignment = self.compute_metrics(cos_sim_mat, eval_alignment, top_k)
        del cos_sim_mat
        return cos_metrics, pred_alignment

    def predict_alignment(self, embed1: np.ndarray, embed2: np.ndarray):
        sim_mtx = self.csls_sim(embed1, embed2, k=10)

        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            pred_list = []
            for cursor in trange(0, total_size, self.batch_size, desc="predict alignment"):
                if isinstance(sim_mtx, np.ndarray):
                    sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                    sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                else:
                    sub_sim_mtx = sim_mtx[cursor: cursor + self.batch_size].to(self.device)

                pred_ranking = torch.argsort(sub_sim_mtx, dim=1, descending=True)
                pred_list.append(pred_ranking[:, 0].cpu().numpy())
        pred_arr = np.concatenate(pred_list, axis=0)
        pred_alignment = np.stack([np.arange(len(embed1)), pred_arr], axis=1)
        return pred_alignment

    def csls_sim(self, embed1: np.ndarray, embed2: np.ndarray, k=10):
        t1 = time.time()
        sim_mat = self.cosine_sim(embed1, embed2)
        if k <= 0:
            print("k = 0")
            return sim_mat
        csls1 = self.CSLS_thr(sim_mat, k)
        csls2 = self.CSLS_thr(sim_mat.T, k)

        # csls_sim_mat = 2 * sim_mat.T - csls1
        # csls_sim_mat = csls_sim_mat.T - csls2
        csls_sim_mat = self.compute_csls(sim_mat, csls1, csls2)
        # del sim_mat
        # gc.collect()
        t2 = time.time()
        print(f"sim handler spends time: {t2 - t1}s")
        return csls_sim_mat

    def compute_csls(self, sim_mtx: np.ndarray, row_thr:np.ndarray, col_thr:np.ndarray):
        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            col_thr = torch.tensor(col_thr, device=self.device).unsqueeze(dim=0)
            csls_sim_mtx = np.empty_like(sim_mtx)
            csls_sim_mtx_list = []
            for cursor in trange(0, total_size, self.batch_size, desc="csls metrix"):
                sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                sub_row_thr = row_thr[cursor: cursor+self.batch_size]
                sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                sub_row_thr = torch.tensor(sub_row_thr, device=self.device)
                sub_sim_mtx = 2*sub_sim_mtx - sub_row_thr.unsqueeze(dim=1) - col_thr
                # csls_sim_mtx_list.append(sub_sim_mtx.cpu().numpy())
                # csls_sim_mtx[cursor: cursor+self.batch_size] = sub_sim_mtx.cpu().numpy()
                sim_mtx[cursor: cursor+self.batch_size] = sub_sim_mtx.cpu().numpy()
        # csls_sim_mtx = np.concatenate(csls_sim_mtx_list, axis=0)
        # return csls_sim_mtx
        return sim_mtx



    def cosine_sim(self, embed1: np.ndarray, embed2: np.ndarray):
        with torch.no_grad():
            total_size = embed1.shape[0]
            embed2 = torch.tensor(embed2, device=self.device).t()
            sim_mtx= np.empty(shape=(embed1.shape[0], embed2.shape[1]), dtype=np.float32)
            sim_mtx_list = []
            for cursor in trange(0, total_size, self.batch_size, desc="cosine matrix"):
                sub_embed1 = embed1[cursor: cursor + self.batch_size]
                sub_embed1 = torch.tensor(sub_embed1, device=self.device)
                sub_sim_mtx = torch.matmul(sub_embed1, embed2)
                # sim_mtx_list.append(sub_sim_mtx.cpu().numpy())
                sim_mtx[cursor: cursor + self.batch_size] = sub_sim_mtx.cpu().numpy()
            # sim_mtx = np.concatenate(sim_mtx_list, axis=0)
        return sim_mtx

    def CSLS_thr(self, sim_mtx: np.ndarray, k=10):
        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            sim_value_list = []
            for cursor in trange(0, total_size, self.batch_size, desc="csls thr"):
                sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                nearest_k, _ = torch.topk(sub_sim_mtx, dim=1, k=k, largest=True, sorted=False)
                sim_values = nearest_k.mean(dim=1, keepdim=False)
                sim_value_list.append(sim_values.cpu().numpy())
            sim_values = np.concatenate(sim_value_list, axis=0)
        return sim_values

    def compute_metrics(self, sim_mtx, eval_alignment, top_k=(1, 5, 10, 50)):
        eval_alignment_arr = np.array(eval_alignment, dtype=np.int)
        # sim_mtx = sim_mtx[eval_alignment_arr[:, 0]][:, eval_alignment_arr[:, 1]]

        eval_alignment_tensor = torch.tensor(eval_alignment, dtype=torch.long, device=self.device)

        with torch.no_grad():
            # total_size = eval_alignment_arr.shape[0]
            total_size = sim_mtx.shape[0]
            gold_rank_list = []
            pred_list = []
            # rels = eval_alignment_tensor[:, 1]
            rels = torch.zeros(size=(total_size,), device=self.device, dtype=torch.long)
            rels[eval_alignment_tensor[:, 0]] = eval_alignment_tensor[:, 1]
            for cursor in trange(0, total_size, self.batch_size, desc="metrics"):
                # sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                # sub_sim_mtx = sim_mtx[eval_alignment_arr[:, 0][cursor: cursor+self.batch_size]][:, eval_alignment_arr[:,1]]
                # sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size][:, eval_alignment_arr[:,1]]
                # sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)

                if isinstance(sim_mtx, np.ndarray):
                    sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                    sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)[:, eval_alignment_arr[:,1]]
                else:
                    sub_sim_mtx = sim_mtx[cursor: cursor + self.batch_size].to(self.device)[:, eval_alignment_arr[:,1]]

                sorted_idxes = torch.argsort(sub_sim_mtx, dim=1, descending=True)
                pred_ranking = eval_alignment_tensor[:, 1][sorted_idxes]
                pred_list.append(pred_ranking[:, 0].cpu().numpy())
                sub_rels = rels[cursor: cursor+self.batch_size].unsqueeze(dim=1)
                k, v = (pred_ranking == sub_rels).nonzero(as_tuple=True)
                gold_idxes = torch.zeros(size=(pred_ranking.shape[0],), dtype=torch.long, device=pred_ranking.device)
                gold_idxes[k] = v
                gold_rank_list.append(gold_idxes.cpu().numpy())
        gold_rank_arr = np.concatenate(gold_rank_list, axis=0) + 1
        gold_rank_arr = gold_rank_arr[eval_alignment_arr[:, 0]]
        pred_arr = np.concatenate(pred_list, axis=0)
        pred_arr = pred_arr[eval_alignment_arr[:, 0]]
        pred_alignment = np.stack([eval_alignment_arr[:, 0], pred_arr], axis=1)
        mean_rank = np.mean(gold_rank_arr)
        mrr = np.mean(1.0 / gold_rank_arr)
        metrics = {"mr": float(mean_rank), "mrr": float(mrr)}
        for k in top_k:
            recall_k = np.mean((gold_rank_arr <= k).astype(np.float32))
            metrics[f"recall@{k}"] = float(recall_k)
        return metrics, pred_alignment


