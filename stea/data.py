# -*- coding: utf-8 -*-

import os
import numpy as np


def read_tab_lines(fn):
    with open(fn) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        tuple_list = []
        for line in lines:
            t = line.split("\t")
            tuple_list.append(t)
    return tuple_list


def write_tab_lines(tuple_list, fn):
    with open(fn, "w+") as file:
        for tup in tuple_list:
            file.write("\t".join(tup) + "\n")


def load_data_for_kg(data_dir, kgid, add_inverse_edge=False):
    ent_id_uri_list = read_tab_lines(os.path.join(data_dir, f"{kgid}_entity_id2uri.txt"))
    rel_id_uri_list = read_tab_lines(os.path.join(data_dir, f"{kgid}_relation_id2uri.txt"))
    triple_rel_list = read_tab_lines(os.path.join(data_dir, f"{kgid}_triple_rel.txt"))

    ent_id_uri_list = [(int(id), uri) for id, uri in ent_id_uri_list]
    rel_id_uri_list = [(int(id), uri) for id, uri in rel_id_uri_list]
    triple_rel_list = [(int(ent1_id), int(rel_id), int(ent2_id)) for ent1_id, rel_id, ent2_id in triple_rel_list]

    if add_inverse_edge:
        # add inverse relations
        max_rel_id = int(np.array([id for id, uri in rel_id_uri_list]).max())
        inverse_rel_uri_list = []
        rel_to_invrel_map = dict()
        for id, uri in rel_id_uri_list:
            max_rel_id += 1
            inverse_rel_uri_list.append((max_rel_id, f"inv_{uri}"))
            rel_to_invrel_map[id] = max_rel_id
        rel_id_uri_list.extend(inverse_rel_uri_list)

        # add inverse triples
        inverse_triple_rel_list = [(t,rel_to_invrel_map[r], h) for h, r, t in triple_rel_list]
        triple_rel_list.extend(inverse_triple_rel_list)

    if os.path.exists(os.path.join(data_dir, f"{kgid}_attribute_id2uri.txt")):
        attr_id_uri_list = read_tab_lines(os.path.join(data_dir, f"{kgid}_attribute_id2uri.txt"))
        attr_id_uri_list = [(int(id), uri) for id, uri in attr_id_uri_list]
    else:
        attr_id_uri_list = None

    if os.path.exists(os.path.join(data_dir, f"{kgid}_triple_attr.txt")):
        triple_attr_list = read_tab_lines(os.path.join(data_dir, f"{kgid}_triple_attr.txt"))
        triple_attr_list = [[int(triple[0]), int(triple[1])]+triple[2:] for triple in triple_attr_list]
    else:
        triple_attr_list = None
    return ent_id_uri_list, rel_id_uri_list, attr_id_uri_list, triple_rel_list, triple_attr_list


def load_alignment(fn):
    if os.path.exists(fn):
        alignment = read_tab_lines(fn)
        alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
    else:
        alignment=[]
    return alignment
def load_alignment_inv(fn):
    if os.path.exists(fn):
        alignment = read_tab_lines(fn)
        alignment = [(int(ent2_id), int(ent1_id)) for ent1_id, ent2_id in alignment]
    else:
        alignment=[]
    return alignment


class KG:
    def __init__(self, data_dir, kgid, add_inverse_edge=False):
        ent_id_uri_list, rel_id_uri_list, attr_id_uri_list, triple_rel_list, triple_attr_list = load_data_for_kg(data_dir, kgid, add_inverse_edge)

        self.num_entity = len(ent_id_uri_list)
        self.num_relation = len(rel_id_uri_list)
        self.triple_rel_list = triple_rel_list

        self.entities = np.arange(0, self.num_entity)

        # self.num_attr = len(attr_id_uri_list)
        self.triple_attr_list = triple_attr_list


class RawDataset4GCNModel:
    def __init__(self, data_dir):
        if data_dir.endswith("/"):
            data_dir = os.path.dirname(data_dir)
        dir_name = os.path.basename(data_dir)
        kg1_id, kg2_id = dir_name.split("_")
        self.kg1 = KG(data_dir, kg1_id, add_inverse_edge=True)
        self.kg2 = KG(data_dir, kg2_id, add_inverse_edge=True)
        self.kg1.triple_rel_list = [(h, t, r) for h, r, t in self.kg1.triple_rel_list]
        self.kg2.triple_rel_list = [(h, t, r) for h, r, t in self.kg2.triple_rel_list]

        train_alignment = load_alignment(os.path.join(data_dir, "train_alignment.txt"))
        train_num = len(train_alignment)
        self.train_alignment = train_alignment[:int(0.8*train_num)]
        self.valid_alignment = train_alignment[int(0.8 * train_num):]
        self.test_alignment = load_alignment(os.path.join(data_dir, "test_alignment.txt"))

        train_alignment_arr = np.array(self.train_alignment)
        valid_alignment_arr = np.array(self.valid_alignment)
        self.kg1_labelled_entities = np.concatenate([train_alignment_arr[:, 0], valid_alignment_arr[:, 0]], axis=0)
        self.kg1_unlabelled_entities = np.array(list(set(self.kg1.entities).difference(set(self.kg1_labelled_entities))))
        self.kg2_labelled_entities = np.concatenate([train_alignment_arr[:, 1], valid_alignment_arr[:, 1]], axis=0)
        self.kg2_unlabelled_entities = np.array(list(set(self.kg2.entities).difference(set(self.kg2_labelled_entities))))




def prepare_data_for_rrea(data_dir, percent_of_training_data):
    with open(os.path.join(data_dir, "ref_ent_ids")) as file:
        lines = file.read().strip().split("\n")
    np.random.shuffle(lines)
    train_num = int(len(lines) * percent_of_training_data)
    train_lines = lines[:train_num]
    test_lines = lines[train_num:]
    with open(os.path.join(data_dir, "ref_ent_ids_train"), "w+") as file:
        file.write("\n".join(train_lines))
    with open(os.path.join(data_dir, "ref_ent_ids_test"), "w+") as file:
        file.write("\n".join(test_lines))
    if not os.path.exists(os.path.join(data_dir,"acc")):
        os.mkdir(os.path.join(data_dir,"acc"))
    if os.path.isfile(os.path.join(data_dir,"acc", "selftraining_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "selftraining_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "overlap_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "overlap_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "st_non_acc.txt")):
        os.remove(os.path.join(data_dir,"acc","st_non_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "paris_non_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "paris_non_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "paris_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "paris_acc.txt"))    
    if os.path.isfile(os.path.join(data_dir,"acc", "st_new_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "st_new_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "o_pair.txt")):
        os.remove(os.path.join(data_dir,"acc", "o_pair.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "rewrite_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "rewrite_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "neural_thr_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "neural_thr_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "neural_eliminated_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "neural_eliminated_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "joint_thr_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "joint_thr_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "joint_eliminated_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "joint_eliminated_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "joint_mn_thr_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "joint_mn_thr_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "neu_mn_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "neu_mn_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "disc_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "disc_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "gen_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "gen_acc.txt"))
    if os.path.isfile(os.path.join(data_dir,"acc", "pseudo_acc.txt")):
        os.remove(os.path.join(data_dir,"acc", "pseudo_acc.txt"))
        
    for i in range(3):
        if os.path.isfile(os.path.join(data_dir,"acc",f"thr_{0.2+0.3*i}_acc.txt")):
            os.remove(os.path.join(data_dir,"acc", f"thr_{0.2+0.3*i}_acc.txt"))
        if os.path.isfile(os.path.join(data_dir,"acc", f"thr_{0.2+0.3*i}_inv_acc.txt")):
            os.remove(os.path.join(data_dir,"acc", f"thr_{0.2+0.3*i}_inv_acc.txt"))
        if os.path.isfile(os.path.join(data_dir,"acc", f"thr_{0.2+0.3*i}_bil_acc.txt")):
            os.remove(os.path.join(data_dir,"acc", f"thr_{0.2+0.3*i}_bil_acc.txt"))
def prepare_data_for_openea(data_dir):
    with open(os.path.join(data_dir, "ent_ids_1")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        tuples = [line.split() for line in lines]
        kg1_ent_id2uri = dict(tuples)

    with open(os.path.join(data_dir, "ent_ids_2")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        tuples = [line.split() for line in lines]
        kg2_ent_id2uri = dict(tuples)

    with open(os.path.join(data_dir, "triples_1")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        kg1_new_lines = []
        for line in lines:
            s, p, o = line.split()
            kg1_new_lines.append(f"{kg1_ent_id2uri[s]}\t{p}\t{kg1_ent_id2uri[o]}")

    with open(os.path.join(data_dir, "triples_2")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        kg2_new_lines = []
        for line in lines:
            s, p, o = line.split()
            kg2_new_lines.append(f"{kg2_ent_id2uri[s]}\t{p}\t{kg2_ent_id2uri[o]}")

    with open(os.path.join(data_dir, "ref_ent_ids")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        new_links = []
        for line in lines:
            e1, e2 = line.split()
            new_links.append(f"{kg1_ent_id2uri[e1]}\t{kg2_ent_id2uri[e2]}")

    with open(os.path.join(data_dir, "ref_ent_ids_train")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        train_new_links = []
        for line in lines:
            e1, e2 = line.split()
            train_new_links.append(f"{kg1_ent_id2uri[e1]}\t{kg2_ent_id2uri[e2]}")

    with open(os.path.join(data_dir, "ref_ent_ids_test")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        test_new_links = []
        for line in lines:
            e1, e2 = line.split()
            test_new_links.append(f"{kg1_ent_id2uri[e1]}\t{kg2_ent_id2uri[e2]}")

    out_dir = os.path.join(data_dir, "openea_format")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "ent_links"), "w+") as file:
        cont = "\n".join(new_links)
        file.write(cont)

    with open(os.path.join(out_dir, "rel_triples_1"), "w+") as file:
        cont = "\n".join(kg1_new_lines)
        file.write(cont)

    with open(os.path.join(out_dir, "rel_triples_2"), "w+") as file:
        cont = "\n".join(kg2_new_lines)
        file.write(cont)

    with open(os.path.join(out_dir, "attr_triples_1"), "w+") as file:
        file.write("")

    with open(os.path.join(out_dir, "attr_triples_2"), "w+") as file:
        file.write("")


    partition_dir = os.path.join(out_dir, "partition")
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    train_num = int(len(train_new_links) * 0.8)
    with open(os.path.join(partition_dir, "train_links"), "w+") as file:
        cont = "\n".join(train_new_links[:train_num])
        file.write(cont)
    with open(os.path.join(partition_dir, "valid_links"), "w+") as file:
        cont = "\n".join(train_new_links[train_num:])
        file.write(cont)
    with open(os.path.join(partition_dir, "test_links"), "w+") as file:
        cont = "\n".join(test_new_links)
        file.write(cont)


def update_openea_enhanced_train_data(data_dir):
    with open(os.path.join(data_dir, "ent_ids_1")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        tuples = [line.split() for line in lines]
        kg1_ent_id2uri = dict(tuples)

    with open(os.path.join(data_dir, "ent_ids_2")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        tuples = [line.split() for line in lines]
        kg2_ent_id2uri = dict(tuples)

    partition_dir = os.path.join(data_dir, "openea_format", "partition")
    with open(os.path.join(data_dir, "ref_ent_ids_train_enhanced")) as file:
        cont = file.read().strip()
        lines = cont.split("\n")
        train_new_links = []
        for line in lines:
            e1, e2 = line.split()
            train_new_links.append(f"{kg1_ent_id2uri[e1]}\t{kg2_ent_id2uri[e2]}")

    train_num = int(len(train_new_links) * 0.8)
    with open(os.path.join(partition_dir, "train_links"), "w+") as file:
        cont = "\n".join(train_new_links[:train_num])
        file.write(cont)
    with open(os.path.join(partition_dir, "valid_links"), "w+") as file:
        cont = "\n".join(train_new_links[train_num:])
        file.write(cont)


def prepare_data_for_joint_distr(data_dir, data_name):
    kg1_id, kg2_id = data_name.split("_")
    # print(kg1_id,kg2_id)
    ent1_oldid2newid_map = dict()
    with open(os.path.join(data_dir, "ent_ids_1"),encoding="utf-8") as file:
        lines = file.read().strip().split("\n")
        ent1_new_lines = []
        for idx, line in enumerate(lines):
            old_ent1, _ = line.split()
            ent1_new_lines.append(f"{idx}\t{old_ent1}")
            ent1_oldid2newid_map[old_ent1] = str(idx)
        new_cont = "\n".join(ent1_new_lines)
    with open(os.path.join(data_dir, f"{kg1_id}_entity_id2uri.txt"), "w+") as file:
        file.write(new_cont)

    ent2_oldid2newid_map = dict()
    with open(os.path.join(data_dir, "ent_ids_2"),encoding="utf-8") as file:
        lines = file.read().strip().split("\n")
        ent2_new_lines = []
        for idx, line in enumerate(lines):
            old_ent2, _ = line.split()
            ent2_new_lines.append(f"{idx}\t{old_ent2}")
            ent2_oldid2newid_map[old_ent2] = str(idx)
        new_cont = "\n".join(ent2_new_lines)
    with open(os.path.join(data_dir, f"{kg2_id}_entity_id2uri.txt"), "w+") as file:
        file.write(new_cont)

    with open(os.path.join(data_dir, "triples_1")) as file:
        lines = file.read().strip().split("\n")
        rel1_list = []
        for line in lines:
            h,r,t = line.split()
            rel1_list.append(int(r))
        rel1_list = sorted(list(set(rel1_list)))
        rel1_oldid2newid_map = dict()
        rel1_lines = []
        for idx, relid in enumerate(rel1_list):
            rel1_oldid2newid_map[relid] = idx
            rel1_lines.append(f"{idx}\t{relid}")
        triple1_new_lines = []
        for line in lines:
            h, r, t = line.split()
            triple1_new_lines.append(f"{ent1_oldid2newid_map[h]}\t{rel1_oldid2newid_map[int(r)]}\t{ent1_oldid2newid_map[t]}")
    with open(os.path.join(data_dir, f"{kg1_id}_triple_rel.txt"), "w+") as file:
        new_cont = "\n".join(triple1_new_lines)
        file.write(new_cont)
    with open(os.path.join(data_dir, f"{kg1_id}_relation_id2uri.txt"), "w+") as file:
        file.write("\n".join(rel1_lines))

    with open(os.path.join(data_dir, "triples_2")) as file:
        lines = file.read().strip().split("\n")
        rel2_list = []
        for line in lines:
            h,r,t = line.split()
            rel2_list.append(int(r))
        rel2_list = sorted(list(set(rel2_list)))
        rel2_oldid2newid_map = dict()
        rel2_lines = []
        for idx, relid in enumerate(rel2_list):
            rel2_oldid2newid_map[relid] = idx
            rel2_lines.append(f"{idx}\t{relid}")
        triple2_new_lines = []
        for line in lines:
            h, r, t = line.split()
            triple2_new_lines.append(f"{ent2_oldid2newid_map[h]}\t{rel2_oldid2newid_map[int(r)]}\t{ent2_oldid2newid_map[t]}")
        new_cont = "\n".join(triple2_new_lines)
    with open(os.path.join(data_dir, f"{kg2_id}_triple_rel.txt"), "w+") as file:
        file.write(new_cont)
    with open(os.path.join(data_dir, f"{kg2_id}_relation_id2uri.txt"), "w+") as file:
        file.write("\n".join(rel2_lines))

    with open(os.path.join(data_dir, "ref_ent_ids_train")) as file:
        lines = file.read().strip().split("\n")
        train_new_lines = []
        for line in lines:
            ent1, ent2 = line.split()
            new_line = f"{ent1_oldid2newid_map[ent1]}\t{ent2_oldid2newid_map[ent2]}"
            train_new_lines.append(new_line)
    with open(os.path.join(data_dir, "train_alignment.txt"), "w+") as file:
        file.write("\n".join(train_new_lines))

    with open(os.path.join(data_dir, "ref_ent_ids_test")) as file:
        lines = file.read().strip().split("\n")
        test_new_lines = []
        for line in lines:
            ent1, ent2 = line.split()
            new_line = f"{ent1_oldid2newid_map[ent1]}\t{ent2_oldid2newid_map[ent2]}"
            test_new_lines.append(new_line)
    with open(os.path.join(data_dir, "test_alignment.txt"), "w+") as file:
        file.write("\n".join(test_new_lines))

    with open(os.path.join(data_dir, "alignment_of_entity.txt"), "w+") as file:
        file.write("\n".join(train_new_lines + test_new_lines))



class EMEAData:
    def __init__(self, data_dir, data_name):
        self.data_dir = data_dir
        self.data_name = data_name

    def load_alignment(self, name):
        alignment = load_alignment(os.path.join(self.data_dir, name))
        return alignment

    def load_kgs(self):
        kg1id, kg2id = self.data_name.split("_")
        kg1 = KG(data_dir=self.data_dir, kgid=kg1id, add_inverse_edge=True)
        kg2 = KG(data_dir=self.data_dir, kgid=kg2id, add_inverse_edge=True)
        return kg1, kg2

    def old2new_entity_id_map(self):
        kg1id, kg2id = self.data_name.split("_")
        kg1_new_old_id_pairs = load_alignment(os.path.join(self.data_dir, f"{kg1id}_entity_id2uri.txt"))
        kg2_new_old_id_pairs = load_alignment(os.path.join(self.data_dir, f"{kg2id}_entity_id2uri.txt"))
        kg1_old_new_id_pairs = [(oldid, newid) for newid, oldid in kg1_new_old_id_pairs]
        kg2_old_new_id_pairs = [(oldid, newid) for newid, oldid in kg2_new_old_id_pairs]
        kg1_old2new_map = dict(kg1_old_new_id_pairs)
        kg2_old2new_map = dict(kg2_old_new_id_pairs)
        return kg1_old2new_map, kg2_old2new_map


