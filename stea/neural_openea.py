# -*- coding: utf-8 -*-


from stea.emea_framework import NeuralEAModule
from stea.conf import Config
import os
import subprocess
from stea.simi_to_prob import SimiToProbModule
import numpy as np
from stea.RREA.CSLS_torch import Evaluator
import torch
from stea.data import load_alignment, load_alignment_inv
import json
from stea.data import update_openea_enhanced_train_data
import itertools
# from graph_tool.all import Graph, max_cardinality_matching  # necessary


class OpenEAModule(NeuralEAModule):
    def __init__(self, conf: Config):
        super(OpenEAModule, self).__init__(conf)

    # def prepare_data(self):
    #     pass

    def train_model_with_observed_labels(self):
        cmd_fn = self.conf.py_exe_fn
        script_fn = self.conf.openea_script_fn
        args_str = f"{self.conf.openea_arg_fn} {self.conf.data_dir}/openea_format/ partition {self.conf.output_dir}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.conf.tf_device
        env["PYTHONPATH"] = os.path.join(os.path.dirname(script_fn), "../src/") #+ ";" + env["PYTHONPATH"]
        ret = subprocess.run(cmd_fn + " " + script_fn + " " + args_str, shell=True, env=env)
        if ret.returncode != 0:
            raise Exception("AliNet did not run successfully.")

        # train simi to prob model
        simi_mtx = self.predict_simi()
        print('simi shape',simi_mtx.shape)
        simi2prob_model = SimiToProbModule(conf=self.conf)
        simi2prob_model.train_model(simi_mtx)
        simi2prob_model_inv = SimiToProbModule(conf=self.conf, inv=True)
        simi2prob_model_inv.train_model(simi_mtx)

    def train_model_with_observed_n_latent_labels(self,ite):
        print("########## Enhance Data #########")
        update_openea_enhanced_train_data(self.conf.data_dir)
        self.train_model_with_observed_labels()

    def predict_simi(self):
        ent1_embs, ent2_embs = self.get_embeddings()
        evaluator = Evaluator()
        simi_mtx = evaluator.csls_sim(ent1_embs, ent2_embs, k=10)
        return simi_mtx

    def predict(self):

        simi_mtx = self.predict_simi()
        if self.conf.no_sim2prob=="False":
            prob_mtx = self.convert_simi_to_probs7(simi_mtx)
            prob_mtx_inv = self.convert_simi_to_probs7(simi_mtx,inv=True)
            prob_mtx = torch.tensor(prob_mtx, device=torch.device(self.conf.second_device))
            prob_mtx_inv = torch.tensor(prob_mtx_inv, device=torch.device(self.conf.second_device))
        else:
            prob_mtx=simi_mtx
            ma=np.amax(prob_mtx, axis=1)
            mi=np.amin(prob_mtx, axis=1)
            prob_mtx=prob_mtx-mi.reshape(-1,1)/(ma.reshape(-1,1)-mi.reshape(-1,1))
            prob_mtx_inv=simi_mtx.transpose()
            ma=np.amax(prob_mtx_inv, axis=1)
            mi=np.amin(prob_mtx_inv, axis=1)
            prob_mtx_inv=prob_mtx_inv-mi.reshape(-1,1)/(ma.reshape(-1,1)-mi.reshape(-1,1))
            prob_mtx = torch.tensor(prob_mtx, device=torch.device(self.conf.device))
            prob_mtx_inv = torch.tensor(prob_mtx_inv, device=torch.device(self.conf.device))
        
        del simi_mtx
        return prob_mtx, prob_mtx_inv


    def get_embeddings(self):
        embs = np.load(os.path.join(self.conf.output_dir, "ent_embeds.npy"))
        with open(os.path.join(self.conf.output_dir, "kg1_ent_ids")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            tuples = [line.split() for line in lines]
            tuples = [(uri, int(id)) for uri, id in tuples]
            kg1_ent_uri2id_map = dict(tuples)
        with open(os.path.join(self.conf.output_dir, "kg2_ent_ids")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            tuples = [line.split() for line in lines]
            tuples = [(uri, int(id)) for uri, id in tuples]
            kg2_ent_uri2id_map = dict(tuples)
        with open(os.path.join(self.conf.data_dir, "ent_ids_1")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            uri_list = [line.split()[1] for line in lines]
            kg1_ent_id_list = [kg1_ent_uri2id_map[uri] for uri in uri_list]
        with open(os.path.join(self.conf.data_dir, "ent_ids_2")) as file:
            cont = file.read().strip()
            lines = cont.split("\n")
            uri_list = [line.split()[1] for line in lines]
            kg2_ent_id_list = [kg2_ent_uri2id_map[uri] for uri in uri_list]

        ent1_ids = np.array(kg1_ent_id_list)
        ent2_ids = np.array(kg2_ent_id_list)
        ent1_embs = embs[ent1_ids]
        ent2_embs = embs[ent2_ids]
        return ent1_embs, ent2_embs

    def get_pred_alignment(self):
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json")) as file:
            obj = json.loads(file.read())
            pred_alignment = obj["pred_alignment_csls"]
        return pred_alignment

    def enhance_latent_labels_with_jointdistr_nochange(self):
        print('enhance_latent_labels_with_jointdistr_nochange')
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        
        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in train_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")

    def enhance_latent_labels_with_jointdistr_neural_one2one(self, neural_sim_mtx: np.ndarray,ratio=0):
        from graph_tool.all import Graph, max_cardinality_matching  # necessary
        print('enhance_latent_labels_with_jointdistr_neural_one2one')
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)


        kg1_entities = load_alignment_inv(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment_inv(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        
        kg1_oldid_to_newid_map = dict(kg1_entities)
        kg2_oldid_to_newid_map = dict(kg2_entities)
        
        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        train_alignment_dict=dict(train_alignment)
        
        pre_labeled_alignment_t=load_alignment(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"))[len(train_alignment):]
        pre_labeled_alignment=[]
        for ent1,ent2 in pre_labeled_alignment_t:
            pre_labeled_alignment.append([kg1_oldid_to_newid_map[ent1],kg2_oldid_to_newid_map[ent2]])
        
        print('ratio',ratio)
        ptrue=0
        pfalse=0
        ptrue_unique=0
        
        new_alignment=[]
        
                
        maxsim=neural_sim_mtx.max()
        print('maxsim',maxsim)
        minsim=min(neural_sim_mtx.max(axis=1))
        print('minsim',minsim)
        minmin=neural_sim_mtx.min()
        print('minmin',minmin)
        # maxprobs={}
        for i in train_alignment:
            new_alignment.append(i)


        x, y = np.where(neural_sim_mtx > ratio)
        potential_aligned_pairs=set(zip(x, y))
        
        k=self.conf.topK
        neighbors = set()
        num = neural_sim_mtx.shape[0]
        for i in range(num):
            rank = np.argpartition(-neural_sim_mtx[i, :], k)
            pairs = [j for j in itertools.product([i], rank[0:k])]
            neighbors |= set(pairs)
            
            
        potential_aligned_pairs &= neighbors
        
        pairs = list(potential_aligned_pairs)
        g = Graph()
        weight_map = g.new_edge_property("float")
        nodes_dict1 = dict()
        nodes_dict2 = dict()
        edges = list()
        for x, y in pairs:
            if x not in nodes_dict1.keys():
                n1 = g.add_vertex()
                nodes_dict1[x] = n1
            if y not in nodes_dict2.keys():
                n2 = g.add_vertex()
                nodes_dict2[y] = n2
            n1 = nodes_dict1.get(x)
            n2 = nodes_dict2.get(y)
            e = g.add_edge(n1, n2)
            edges.append(e)
            weight_map[g.edge(n1, n2)] = neural_sim_mtx[x, y]
        print("graph via graph_tool", g)
        res = max_cardinality_matching(g, heuristic=True, weight=weight_map, minimize=False)
        edge_index = np.where(res.get_array() == 1)[0].tolist()
        curr_labeled_alignment = set()
        for index in edge_index:
            curr_labeled_alignment.add(pairs[index])
        
        
        
        labeled_alignment_dict = dict(pre_labeled_alignment)
        for i, j in curr_labeled_alignment:
            if i in train_alignment_dict or j in train_alignment_dict:
                continue
            if i in labeled_alignment_dict.keys():
                pre_j = labeled_alignment_dict.get(i)
                pre_sim = neural_sim_mtx[i, pre_j]
                new_sim = neural_sim_mtx[i, j]
                if new_sim >= pre_sim:
                    labeled_alignment_dict[i] = j
            else:
                labeled_alignment_dict[i] = j
        pre_labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))


        labeled_alignment_dict = dict()
        for i, j in pre_labeled_alignment:
            i_set = labeled_alignment_dict.get(j, set())
            i_set.add(i)
            labeled_alignment_dict[j] = i_set
        for j, i_set in labeled_alignment_dict.items():
            if len(i_set) == 1:
                for i in i_set:
                    new_alignment.append([i, j])
                    if i==j:
                        ptrue+=1
                        if i not in train_alignment_dict:
                            ptrue_unique+=1
                    else:
                        pfalse+=1
            else:
                max_i = -1
                max_sim = -10
                for i in i_set:
                    if neural_sim_mtx[i, j] > max_sim:
                        max_sim = neural_sim_mtx[i, j]
                        max_i = i
                new_alignment.append([max_i, j])
                if max_i==j:
                    ptrue+=1
                else:
                    pfalse+=1        
        
        print('size of new-alignment',len(new_alignment))        
        rec_num = ptrue+pfalse
        if rec_num == 0:
            pacc = 0
        else:
            pacc=ptrue/(ptrue+pfalse)
        precall=ptrue_unique/len(test_alignment)
        if pacc+precall == 0:
            pf1 = 0
        else:
            pf1=2*pacc*precall/(pacc+precall)

            print("neural one2one acc" , pacc)
        
        with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
            file.write(f"{pacc}\t{precall}\t{pf1}\n")

        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")

    def enhance_latent_labels_with_jointdistr_neural_only(self, neural_sim_mtx: np.ndarray,ratio=0,mn=False):
        print('enhance_latent_labels_with_jointdistr_neural_only')
        mn=mn
        direction=self.conf.direction
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        
        print('ratio',ratio)
        ptrue=0
        pfalse=0
        ptrue_unique=0
        new_alignment=[]
        
        d={}
        # maxprobs={}
        for i in train_alignment:
            new_alignment.append(i)

        # maxsim=neural_sim_mtx.max()
        # print('maxsim',maxsim)
        # minsim=min(neural_sim_mtx.max(axis=1))
        # print('minsim',minsim)
        # minmin=neural_sim_mtx.min()
        # print('minmin',minmin)
        if mn:
            for ent1, ent2 in test_alignment:
                tmp_idx = np.argmax(neural_sim_mtx[ent1])
                tmp_idx_inv=np.argmax(neural_sim_mtx[:,tmp_idx])
                if ent1==tmp_idx_inv and neural_sim_mtx[ent1,tmp_idx]>ratio:
                    if ent1==tmp_idx:
                        ptrue+=1
                        ptrue_unique+=1
                    else:
                        pfalse+=1
                        
                    new_alignment.append([ent1,tmp_idx])
                        
        else:
            if direction=="source" or direction=="both":
                for ent1, ent2 in test_alignment:
                    tmp_idx = np.argmax(neural_sim_mtx[ent1])

                    if neural_sim_mtx[ent1,tmp_idx]>ratio:
                        if ent1==tmp_idx:
                            ptrue+=1
                            if ent1 not in d:
                                d[ent1]=0
                                ptrue_unique+=1
                        else:
                            pfalse+=1
                            
                        new_alignment.append([ent1,tmp_idx])
            if direction=="target" or direction=="both":
                for ent1, ent2 in test_alignment:
                    tmp_idx = np.argmax(neural_sim_mtx[:,ent2])

                    if neural_sim_mtx[tmp_idx,ent2]>ratio:
                        if ent2==tmp_idx:
                            ptrue+=1
                            if ent2 not in d:
                                d[ent2]=0
                                ptrue_unique+=1
                        else:
                            pfalse+=1
                            
                        new_alignment.append([tmp_idx,ent2])
        print('size of new-alignment',len(new_alignment))        
        rec_num = ptrue+pfalse
        if rec_num == 0:
            pacc = 0
        else:
            pacc=ptrue/(ptrue+pfalse)
        precall=ptrue_unique/len(test_alignment)
        if pacc+precall == 0:
            pf1 = 0
        else:
            pf1=2*pacc*precall/(pacc+precall)
        if mn:
            print("neural threshold & mutual nearest acc" , pacc)
        else:
            print("neural threshold acc" , pacc)
        
        with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
            file.write(f"{pacc}\t{precall}\t{pf1}\n")

        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")

    
    def enhance_latent_labels_with_jointdistr_joint_distr_mn(self,improved_candi_probs: torch.Tensor, improved_candi_probs_inv: torch.Tensor, ratio=0):
        print('enhance_latent_labels_with_jointdistr_joint_distr_mn')
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        
        ratio=ratio
        print('ratio',ratio)
        ptrue=0
        pfalse=0
        
        new_alignment=[]
        
        
        # maxprobs={}
        for i in train_alignment:
            new_alignment.append(i)
        
        with torch.no_grad():
            for ent1, ent2 in test_alignment:
                probs = improved_candi_probs[ent1]
                prob,tmp_idx = torch.max(probs, dim=-1, keepdim=False)
                prob=prob.cpu().item()
                tmp_idx=tmp_idx.cpu().item()
                probs_inv=improved_candi_probs_inv[tmp_idx]
                prob_inv,tmp_idx_inv=torch.max(probs_inv, dim=-1, keepdim=False)
                prob_inv=prob_inv.cpu().item()
                tmp_idx_inv=tmp_idx_inv.cpu().item()

                if ent1==tmp_idx_inv:
                    if prob>ratio and prob_inv>ratio:
                        if ent1==tmp_idx:
                            ptrue+=1
                        else:
                            pfalse+=1
                        new_alignment.append([ent1,tmp_idx])

        
        rec_num = ptrue+pfalse
        if rec_num == 0:
            pacc = 0
        else:
            pacc=ptrue/(ptrue+pfalse)
        precall=ptrue/len(test_alignment)
        if pacc+precall == 0:
            pf1 = 0
        else:
            pf1=2*pacc*precall/(pacc+precall)
        print("joint distr threshold & mutual nearest acc" , pacc)
        

        with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
            file.write(f"{pacc}\t{precall}\t{pf1}\n")

        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")

    def enhance_latent_labels_with_jointdistr_joint_distr_mn_pseudo(self,improved_candi_probs: torch.Tensor, improved_candi_probs_inv: torch.Tensor, ratio=0,pseudo=0.05):
        print('enhance_latent_labels_with_jointdistr_joint_distr_mn_pseudo')
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        
        ratio=ratio
        print('ratio',ratio)
        pseudo=pseudo
        ptrue=0
        eliminated_true=0
        pfalse=0
        
        new_alignment=[]
        e_pairs=[]
        e_num=0
        
        pairs_for_pseudo=[]
        pstrue=0
        psfalse=0
        
        ent1_dict={}
        ent2_dict={}
        
        gen_pairs=[]
        gen_num=0


        # maxprobs={}
        for i in train_alignment:
            new_alignment.append(i)
            ent1_dict[i[0]]=1
            ent2_dict[i[1]]=1
            
        
        with torch.no_grad():
            for ent1, ent2 in test_alignment:
                if ent1 not in ent1_dict and ent2 not in ent2_dict:
                    probs = improved_candi_probs[ent1]
                    prob,tmp_idx = torch.max(probs, dim=-1, keepdim=False)
                    prob=prob.cpu().item()
                    tmp_idx=tmp_idx.cpu().item()
                    probs_inv=improved_candi_probs_inv[tmp_idx]
                    prob_inv,tmp_idx_inv=torch.max(probs_inv, dim=-1, keepdim=False)
                    prob_inv=prob_inv.cpu().item()
                    tmp_idx_inv=tmp_idx_inv.cpu().item()
                    if ent1==tmp_idx_inv:
                        if prob>ratio and prob_inv>ratio:
                            if ent1==tmp_idx:
                                ptrue+=1
                            else:
                                pfalse+=1
                            new_alignment.append([ent1,tmp_idx])
                            gen_pairs.append([ent1,tmp_idx])
                            gen_num+=1
                            pairs_for_pseudo.append([[ent1,tmp_idx],prob+prob_inv])
                        else:
                            if ent1==tmp_idx:
                                eliminated_true+=1
                            e_pairs.append([ent1,tmp_idx])
                            e_num+=1
        
        print('size of new-alignment',len(new_alignment))        
        pacc=ptrue/(ptrue+pfalse)
        print("joint distr threshold & mutual nearest acc" , pacc)
        if e_num>0:
            eliminated_acc=eliminated_true/e_num
        else:
            eliminated_acc=0
        print("joint distr mutual nearest eliminated by threshold acc", eliminated_acc)
        
        pairs_for_pseudo=sorted(pairs_for_pseudo,key=lambda x:x[1],reverse=True)
        length=round(len(pairs_for_pseudo)*pseudo)+1
        
        
        print("pseudo pairs",length)
        pairs_for_pseudo=pairs_for_pseudo[:length]
        for pair,_ in pairs_for_pseudo:
            if pair[0]==pair[1]:
                pstrue+=1
            else:
                psfalse+=1
            train_alignment.append(pair)
        

        psacc=pstrue/(pstrue+psfalse)
        print("pseudo acc",psacc)
            
        with open(os.path.join(self.conf.data_dir, "train_alignment.txt"), "w+") as file:
            for ent1, ent2 in train_alignment:
                file.write(f"{ent1}\t{ent2}\n")  
        
        with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
            file.write(f"{pacc}\t{gen_num}\n")
            
        with open(os.path.join(self.conf.data_dir,"acc", "pseudo_acc.txt"), "a") as file:
            file.write(f"{psacc}\t{length}\n")

        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")
        with open(os.path.join(self.conf.output_dir, "generated_pairs"), "w+") as file:
            for ent1, ent2 in gen_pairs:
                file.write(f"{ent1}\t{ent2}\n")
        
        with open(os.path.join(self.conf.output_dir, "pseudo_pairs"), "w+") as file:
            for [ent1, ent2],_ in pairs_for_pseudo:
                file.write(f"{ent1}\t{ent2}\n")


    def enhance_latent_labels_with_jointdistr_double_mn(self,improved_candi_probs: torch.Tensor, improved_candi_probs_inv: torch.Tensor,
                                                        neural_sim_mtx: np.ndarray, ratio=0, pri=True,only=False):
        print('enhance_latent_labels_with_jointdistr_double_mn')
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        
        ratio=ratio
        print('joint_thr',ratio)
        joint_true=0
        joint_false=0
        neu_true=0
        neu_false=0
        
        new_alignment=[]
        
        joint_gen_num=0
        neu_gen_num=0
        
        dic_neu={}
        dic_joint={}
        
        # maxsim=improved_candi_probs.max().cpu().item()
        # print('maxsim',maxsim)
        # minsim=improved_candi_probs.max(axis=1)[0].cpu().min().item()
        # print('minsim',minsim)
        # minmin=improved_candi_probs.min().cpu().item()
        # print('minmin',minmin)    

        # maxsim_inv=improved_candi_probs_inv.max().cpu().item()
        # print('maxsim_inv',maxsim_inv)
        # minsim_inv=improved_candi_probs_inv.max(axis=1)[0].cpu().min().item()
        # print('minsim_inv',minsim_inv)
        # minmin_inv=improved_candi_probs_inv.min().cpu().item()
        # print('minmin_inv',minmin_inv)   
        # maxprobs={}
        for i in train_alignment:
            new_alignment.append(i)
            dic_neu[i[0]]=i[1]
            dic_joint[i[0]]=i[1]
        
        with torch.no_grad():
            for ent1, ent2 in test_alignment:
                probs = improved_candi_probs[ent1]
                prob,tmp_idx = torch.max(probs, dim=-1, keepdim=False)
                prob=prob.cpu().item()
                tmp_idx=tmp_idx.cpu().item()
                probs_inv=improved_candi_probs_inv[tmp_idx]
                prob_inv,tmp_idx_inv=torch.max(probs_inv, dim=-1, keepdim=False)
                prob_inv=prob_inv.cpu().item()
                tmp_idx_inv=tmp_idx_inv.cpu().item()
                if ent1==tmp_idx_inv:
                    if prob>ratio and prob_inv>ratio:
                        if ent1==tmp_idx:
                            joint_true+=1
                        else:
                            joint_false+=1
                        dic_joint[ent1]=tmp_idx
                        # joint_gen_pairs.append([ent1,tmp_idx])
                        joint_gen_num+=1

                
                tmp_idx = np.argmax(neural_sim_mtx[ent1])
                tmp_idx_inv=np.argmax(neural_sim_mtx[:,tmp_idx])
                if ent1==tmp_idx_inv:
                    if ent1==tmp_idx:
                        neu_true+=1
                    else:
                        neu_false+=1
                    dic_neu[ent1]=tmp_idx
                    # neu_gen_pairs.append([ent1,tmp_idx])
                    neu_gen_num+=1
            
           
        
        print("joint distr threshold & mutual nearest num" , joint_gen_num)
        joint_acc=joint_true/(joint_true+joint_false)
        print("joint distr threshold & mutual nearest acc" , joint_acc)
        print("neu mutual nearest num" , neu_gen_num)
        neu_acc=neu_true/(neu_true+neu_false)
        print("neu mutual nearest acc" , neu_acc)
        


        
        ptrue=0
        pfalse=0
        if pri:
            print('prioritize joint')
            dic_high,dic_low=dic_joint,dic_neu
        else:
            print('prioritize neu')
            dic_high,dic_low=dic_neu,dic_joint
        for i in dic_high:
            j=dic_high[i]
            new_alignment.append([i,j])
            if i==j:
                ptrue+=1
            else:
                pfalse+=1
                
        for i in dic_low:
            if i not in dic_high:
                j=dic_low[i]
                new_alignment.append([i,j])
                if i==j:
                    ptrue+=1
                else:pfalse+=1

        rec_num = ptrue+pfalse
        if rec_num == 0:
            pacc = 0
        else:
            pacc=ptrue/(ptrue+pfalse)
        precall=ptrue/len(test_alignment)
        if pacc+precall == 0:
            pf1 = 0
        else:
            pf1=2*pacc*precall/(pacc+precall)
        

        with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
            file.write(f"{pacc}\t{precall}\t{pf1}\n")




        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")
 




        print('size of new-alignment',len(new_alignment))     

    def enhance_latent_labels_with_jointdistr_neural_mn(self,improved_candi_probs: torch.Tensor, improved_candi_probs_inv: torch.Tensor,
                                                        neural_sim_mtx: np.ndarray, ratio=0):
        print('enhance_latent_labels_with_jointdistr_neural_mn')
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        
        
        direction=self.conf.direction
        
        new_alignment=[]
        ptrue=0
        ptrue_unique=0
        pfalse=0
        # maxprobs={}
        for i in train_alignment:
            new_alignment.append(i)
            
        tmpprob=improved_candi_probs
            
        # maxsim=tmpprob.max().cpu().item()
        # print('maxsim',maxsim)
        # minsim=tmpprob.max(axis=1)[0].cpu().min().item()
        # print('minsim',minsim)
        # minmin=tmpprob.min().cpu().item()
        # print('minmin',minmin)     

        if direction=="source" or direction=="both":
            with torch.no_grad():
                for ent1,_ in test_alignment:
                    
                    probs,tmp_idx = tmpprob[ent1].max(axis=0)
                    probs=probs.cpu().item()
                    tmp_idx=tmp_idx.cpu().item()
    
                    if probs>ratio:
                        new_alignment.append([ent1,tmp_idx])
    
        del improved_candi_probs
            
        tmpprob=improved_candi_probs_inv
            
        # maxsim=tmpprob.max().cpu().item()
        # print('maxsim_inv',maxsim)
        # minsim=tmpprob.max(axis=1)[0].cpu().min().item()
        # print('minsim_inv',minsim)
        # minmin=tmpprob.min().cpu().item()
        # print('minmin_inv',minmin)     

        
        if direction=="target" or direction=="both":
            with torch.no_grad():
                for _,ent2 in test_alignment:
                    
                    probs,tmp_idx = tmpprob[ent2].max(axis=0)
                    probs=probs.cpu().item()
                    tmp_idx=tmp_idx.cpu().item()
    
                    if probs>ratio:
                        new_alignment.append([tmp_idx,ent2])
        
        
        del improved_candi_probs_inv
        

        news=[]
        d={}
        
        with torch.no_grad():
            for ent1, ent2 in new_alignment:

                tmp_idx = np.argmax(neural_sim_mtx[ent1])
                tmp_idx_inv=np.argmax(neural_sim_mtx[:,tmp_idx])
                if ent1==tmp_idx_inv:
                    if ent1==tmp_idx:
                        if ent1 not in d:
                            d[ent1]=0
                            ptrue_unique+=1
                        ptrue+=1
                    else:
                        pfalse+=1
                    news.append([ent1,ent2])
                    # joint_gen_pairs.append([ent1,tmp_idx])
        
        

        print('size of new-alignment',len(new_alignment))
        rec_num = ptrue+pfalse
        if rec_num == 0:
            pacc = 0
        else:
            pacc=ptrue/(ptrue+pfalse)
        precall=ptrue_unique/len(test_alignment)
        if pacc+precall == 0:
            pf1 = 0
        else:
            pf1=2*pacc*precall/(pacc+precall)
        

        with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
            file.write(f"{pacc}\t{precall}\t{pf1}\n")
        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in news:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")     
        # ratio=ratio
        # print('joint_thr',ratio)
        
        # new_alignment=[]
        
        # gen_pairs=[]
        
        # final_true=0
        
        
        # maxsim=improved_candi_probs.max().cpu().item()
        # print('maxsim',maxsim)
        # minsim=improved_candi_probs.max(axis=1)[0].cpu().min().item()
        # print('minsim',minsim)
        # minmin=improved_candi_probs.min().cpu().item()
        # print('minmin',minmin)    

        # maxsim_inv=improved_candi_probs_inv.max().cpu().item()
        # print('maxsim_inv',maxsim_inv)
        # minsim_inv=improved_candi_probs_inv.max(axis=1)[0].cpu().min().item()
        # print('minsim_inv',minsim_inv)
        # minmin_inv=improved_candi_probs_inv.min().cpu().item()
        # print('minmin_inv',minmin_inv)   
        # # maxprobs={}
        # for i in train_alignment:
        #     new_alignment.append(i)
        
        # with torch.no_grad():
        #     for ent1, ent2 in test_alignment:
        #         tmp_idx = np.argmax(neural_sim_mtx[ent1])
        #         prob=improved_candi_probs[ent1][tmp_idx].cpu().item()
        #         tmp_idx_inv=np.argmax(neural_sim_mtx[:,tmp_idx])
        #         if ent1==tmp_idx_inv:
        #             prob_inv=improved_candi_probs_inv[tmp_idx][ent1].cpu().item()
        #             if prob-minsim>ratio*(maxsim-minsim) and prob_inv-minsim_inv>ratio*(maxsim_inv-minsim_inv):
        #                 if ent1==tmp_idx:
        #                     final_true+=1
        #                 new_alignment.append([ent1,tmp_idx])
        #                 gen_pairs.append([ent1,tmp_idx])
        #                 # joint_gen_pairs.append([ent1,tmp_idx])

        
        # final_num=len(gen_pairs)
        # final_acc=final_true/final_num
        # with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
        #     file.write(f"{final_acc}\t{final_num}\n")




        # with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
        #     for ent1, ent2 in new_alignment:
        #         file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")
        # with open(os.path.join(self.conf.output_dir, "generated_pairs"), "w+") as file:
        #     for ent1, ent2 in gen_pairs:
        #         file.write(f"{ent1}\t{ent2}\n")        




        # print('size of new-alignment',len(new_alignment))     



    def enhance_latent_labels_with_jointdistr_joint_distr_thr(self,improved_candi_probs: torch.Tensor, improved_candi_probs_inv: torch.Tensor,ratio):
        print('enhance_latent_labels_with_jointdistr_joint_distr_thr')
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        

        new_alignment=[]
        ptrue=0
        ptrue_unique=0
        pfalse=0
        
        d={}

        # maxprobs={}
        for i in train_alignment:
            new_alignment.append(i)
        
        
        
        
        direction=self.conf.direction
        
        if direction=="source" or direction=="both":
            tmpprob=improved_candi_probs
                
            # maxsim=tmpprob.max().cpu().item()
            # print('maxsim',maxsim)
            # minsim=tmpprob.max(axis=1)[0].cpu().min().item()
            # print('minsim',minsim)
            # minmin=tmpprob.min().cpu().item()
            # print('minmin',minmin)     
    
            
            with torch.no_grad():
                for ent1,_ in test_alignment:
                    
                    probs,tmp_idx = tmpprob[ent1].max(axis=0)
                    probs=probs.cpu().item()
                    tmp_idx=tmp_idx.cpu().item()
    
                    if probs>ratio:
                        if ent1==tmp_idx:
                            ptrue+=1
                            if ent1 not in d:
                                d[ent1]=0
                                ptrue_unique+=1
                        else:
                            pfalse+=1
                        new_alignment.append([ent1,tmp_idx])

        del improved_candi_probs
        
            
        # maxsim=tmpprob.max().cpu().item()
        # print('maxsim_inv',maxsim)
        # minsim=tmpprob.max(axis=1)[0].cpu().min().item()
        # print('minsim_inv',minsim)
        # minmin=tmpprob.min().cpu().item()
        # print('minmin_inv',minmin)     
        
        if direction=="target" or direction=="both":
        
            tmpprob=improved_candi_probs_inv
            with torch.no_grad():
                for _,ent2 in test_alignment:
                    
                    probs,tmp_idx = tmpprob[ent2].max(axis=0)
                    probs=probs.cpu().item()
                    tmp_idx=tmp_idx.cpu().item()
    
                    if probs>ratio:
                        if ent2==tmp_idx:
                            ptrue+=1
                            if ent2 not in d:
                                d[ent2]=0
                                ptrue_unique+=1
                        else:
                            pfalse+=1
                        new_alignment.append([tmp_idx,ent2])
        
        
        del improved_candi_probs_inv

        print('size of new-alignment',len(new_alignment))
        rec_num = ptrue+pfalse
        if rec_num == 0:
            pacc = 0
        else:
            pacc=ptrue/(ptrue+pfalse)
        precall=ptrue_unique/len(test_alignment)
        if pacc+precall == 0:
            pf1 = 0
        else:
            pf1=2*pacc*precall/(pacc+precall)
        

        with open(os.path.join(self.conf.data_dir,"acc", "gen_acc.txt"), "a") as file:
            file.write(f"{pacc}\t{precall}\t{pf1}\n")
            
            
        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n") 
    def enhance_data2(self):
        pass

    def enhance_data_archive(self, improved_candi_probs: torch.Tensor, improved_candi_probs_inv: torch.Tensor,ratio):       
        kg1id, kg2id = self.conf.data_name.split("_")
        kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_newid_to_oldid_map = dict(kg1_entities)
        kg2_newid_to_oldid_map = dict(kg2_entities)

        train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        


        ptrue=0
        pfalse=0
        poplist=[]
        sampling_num = 1
        sampling_ent2_list = []
        ent1_list=[]
        ent2_list=[]
        
        rewrite_alignment=load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        rewrite_problist=[]
        # maxprobs={}
        for ent1, ent2 in train_alignment:
            sampling_ent2_list.append([ent2]*sampling_num)
            ent1_list.append(ent1)
            ent2_list.append(ent2)
            
        
        position=len(train_alignment)
        pos_dict={}
        
        with torch.no_grad():
            for i, (ent1, ent2) in enumerate(test_alignment):
                if ent1 not in ent1_list and ent2 not in ent2_list:
                    probs = improved_candi_probs[ent1]
                    prob,tmp_idx = torch.max(probs, dim=-1, keepdim=False)
                    prob=prob.cpu().item()
                    tmp_idx=tmp_idx.cpu().numpy()
                    probs_inv=improved_candi_probs_inv[tmp_idx.item()]
                    prob_inv,tmp_idx_inv=torch.max(probs_inv, dim=-1, keepdim=False)
                    prob_inv=prob_inv.cpu().item()
                    tmp_idx_inv=tmp_idx_inv.cpu().numpy()
                    if ent1==tmp_idx_inv.item():
                        if ent1==tmp_idx.item():
                            ptrue+=1
                        else:
                            pfalse+=1
                        ent2_arr = np.array([tmp_idx])
                        sampling_ent2_list.append(ent2_arr)
                        rewrite_problist.append([i,prob+prob_inv])
                        pos_dict[i]=position
                        position+=1
                    else:
                        poplist.append(i)
                else:
                    poplist.append(i)
                    # filtered.append(ent1)
        
        poplist=sorted(poplist,reverse=True)
        for i in poplist:
            test_alignment.pop(i)
        ent1_arr = np.concatenate([np.array(train_alignment)[:, 0], np.array(test_alignment)[:, 0]])
        ent2_mtx = np.stack(sampling_ent2_list, axis=0)
        new_alignment = np.stack([ent1_arr, ent2_mtx[:, 0]], axis=-1).tolist()
        print('size of new-alignment',len(new_alignment))        
        pacc=ptrue/(ptrue+pfalse)
        print("paris acc" , pacc)
        with open(os.path.join(self.conf.data_dir,"acc", "paris_acc.txt"), "a") as file:
            file.write(f"{pacc}\n")
        with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
            for ent1, ent2 in new_alignment:
                file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")        
        
        
        rewrite_problist=sorted(rewrite_problist,key=lambda x:x[1],reverse=False)
        length=int(len(rewrite_problist)*(1-ratio))
        newlength=len(rewrite_problist)-length
        for i in range(length):
            poplist.append(rewrite_problist[i][0])
        print('adding',newlength,'new labels into training set')
        poplist=sorted(poplist,reverse=True)
        pos_poplist=[]
        for i in poplist:
            rewrite_alignment.pop(i)
            if i in pos_dict:
                pos_poplist.append(pos_dict[i])    
        pos_poplist=sorted(pos_poplist,reverse=True)
        for i in pos_poplist:
            sampling_ent2_list.pop(i)         
        ent1_arr = np.concatenate([np.array(train_alignment)[:, 0], np.array(rewrite_alignment)[:, 0]])
        ent2_mtx = np.stack(sampling_ent2_list, axis=0)
        rewrite_alignment=np.stack([ent1_arr, ent2_mtx[:, 0]], axis=-1).tolist()
        with open(os.path.join(self.conf.data_dir, "train_alignment.txt"), "w+") as file:
            for ent1, ent2 in rewrite_alignment:
                file.write(f"{ent1}\t{ent2}\n")

    def enhance_latent_labels_with_jointdistr(self, improved_candi_probs: torch.Tensor, improved_candi_probs_inv: torch.Tensor,
                                              neural_sim_mtx: np.ndarray,candi_mtx,conf):
        
        print(conf.neural_mn)
        if improved_candi_probs is None:
            if conf.neural_one2one == "True":
                self.enhance_latent_labels_with_jointdistr_neural_one2one(neural_sim_mtx=neural_sim_mtx,ratio=conf.neural_thr)
            else:
                self.enhance_latent_labels_with_jointdistr_neural_only(neural_sim_mtx=neural_sim_mtx,ratio=conf.neural_thr,mn=conf.neural_mn)
        else:
            if conf.joint_distr_mn and not conf.neural_mn:
                # if conf.joint_distr_pseudo:
                #     self.enhance_latent_labels_with_jointdistr_joint_distr_mn_pseudo(improved_candi_probs=improved_candi_probs, improved_candi_probs_inv=improved_candi_probs_inv,ratio=conf.joint_distr_thr,pseudo=conf.joint_distr_pseudo)
                # else:
                #     self.enhance_latent_labels_with_jointdistr_joint_distr_mn(improved_candi_probs=improved_candi_probs, improved_candi_probs_inv=improved_candi_probs_inv,ratio=conf.joint_distr_thr)
                self.enhance_latent_labels_with_jointdistr_joint_distr_mn(improved_candi_probs=improved_candi_probs, improved_candi_probs_inv=improved_candi_probs_inv,ratio=conf.joint_distr_thr)
            
            if not conf.joint_distr_mn and not conf.neural_mn:
                self.enhance_latent_labels_with_jointdistr_joint_distr_thr(improved_candi_probs=improved_candi_probs, improved_candi_probs_inv=improved_candi_probs_inv,ratio=conf.joint_distr_thr)
            if conf.joint_distr_mn and conf.neural_mn:
                self.enhance_latent_labels_with_jointdistr_double_mn(improved_candi_probs=improved_candi_probs, improved_candi_probs_inv=improved_candi_probs_inv,
                                                                     neural_sim_mtx=neural_sim_mtx,ratio=conf.joint_distr_thr,pri=conf.joint_distr_pri,only=conf.overlapping_only)
            if  not conf.joint_distr_mn and conf.neural_mn:
                self.enhance_latent_labels_with_jointdistr_neural_mn(improved_candi_probs=improved_candi_probs, improved_candi_probs_inv=improved_candi_probs_inv,
                                                                     neural_sim_mtx=neural_sim_mtx,ratio=conf.joint_distr_thr)
        # else:
        #     kg1id, kg2id = self.conf.data_name.split("_")
        #     kg1_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg1id}_entity_id2uri.txt"))
        #     kg2_entities = load_alignment(os.path.join(self.conf.data_dir, f"{kg2id}_entity_id2uri.txt"))
        #     kg1_newid_to_oldid_map = dict(kg1_entities)
        #     kg2_newid_to_oldid_map = dict(kg2_entities)
    
        #     train_alignment = load_alignment(os.path.join(self.conf.data_dir, "train_alignment.txt"))
        #     test_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))
        #     # idx_range = np.arange(improved_candi_probs.shape[1])
        #     sampling_num = 1
        #     sampling_ent2_list = []
        #     for ent1, ent2 in train_alignment:
        #         sampling_ent2_list.append([ent2]*sampling_num)
        #     with torch.no_grad():
        #         for ent1, _ in test_alignment:
        #             probs = improved_candi_probs[ent1]
        #             # tmp_idx = np.argmax(probs, axis=-1)
        #             tmp_idx = torch.argmax(probs, dim=-1, keepdim=False).cpu().numpy()
        #             ent2_arr = np.array([tmp_idx])
        #             sampling_ent2_list.append(ent2_arr)
        #     ent1_arr = np.concatenate([np.array(train_alignment)[:, 0], np.array(test_alignment)[:, 0]])
        #     ent2_mtx = np.stack(sampling_ent2_list, axis=0)
        #     new_alignment = np.stack([ent1_arr, ent2_mtx[:, 0]], axis=-1).tolist()
        #     with open(os.path.join(self.conf.data_dir, "ref_ent_ids_train_enhanced"), "w+") as file:
        #         for ent1, ent2 in new_alignment:
        #             file.write(f"{kg1_newid_to_oldid_map[ent1]}\t{kg2_newid_to_oldid_map[ent2]}\n")

    def evaluate(self):
        ent1_embs, ent2_embs = self.get_embeddings()

        eval_alignment = load_alignment(os.path.join(self.conf.data_dir, "test_alignment.txt"))

        evaluator = Evaluator()
        cos_test_metrics, cos_test_alignment = evaluator.evaluate_cosine(ent1_embs, ent2_embs, eval_alignment)
        csls_test_metrics, csls_test_alignment = evaluator.evaluate_csls(ent1_embs, ent2_embs, eval_alignment)

        metrics_obj = {"metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
        print("csls metrics: ", csls_test_metrics)
        pred_alignment_obj = {"pred_alignment_csls": csls_test_alignment.tolist(),
                              "pred_alignment_cos": cos_test_alignment.tolist()}
        ranking_list=[]
        neural_sim_mtx = evaluator.csls_sim(ent1_embs, ent2_embs, k=10)
        sim_mtx=-neural_sim_mtx
        for ent1, _ in eval_alignment:
            ranking=np.argsort(np.argsort(sim_mtx[ent1]))[ent1]
            ranking_list.append([ent1,ranking])
            # if ent1==tmp_idx:
            #     ranking=0
            # else:
            #     ranking=torch.sort(torch.sort(-probs).indices).indices[ent1].cpu().item()
            # gen_pairs.append([ent1,ranking])
            
            # if ent1==tmp_idx:
            #     ranking=0
            # else:
            #     ranking=np.argsort(np.argsort(-neural_sim_mtx[ent1]))[ent1]
            # gen_pairs.append([ent1,ranking])        
        with open(os.path.join(self.conf.output_dir, "rankings"), "w+") as file:
            for ent1, ent2 in ranking_list:
                file.write(f"{ent1}\t{ent2}\n")  
        with open(os.path.join(self.conf.output_dir, "metrics.json"), "w+") as file:
            file.write(json.dumps(metrics_obj))
        with open(os.path.join(self.conf.output_dir, "pred_alignment.json"), "w+") as file:
            file.write(json.dumps(pred_alignment_obj))
        with open(os.path.join(self.conf.output_dir, "eval_metrics.json"), "w+") as file:
            file.write(json.dumps(csls_test_metrics))


