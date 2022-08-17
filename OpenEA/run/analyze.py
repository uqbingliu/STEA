# -*- coding: utf-8 -*-

from openea.modules.load.kgs import read_kgs_from_folder
import os
from openea.modules.load.read import read_links, read_relation_triples
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from collections import Counter


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
    return func_map


def subsumption(triple1_list, triple2_list, alignment):
    ent2_to_ent1_map = dict(alignment)
    for ent1, ent2 in alignment:
        ent2_to_ent1_map[ent2] = ent1

    # assign id to each entity
    ent1_list = []
    rel1_list = []
    for head, rel, tail in triple1_list:
        ent1_list.extend([head, tail])
        rel1_list.append(rel)
    ent1_list = list(set(ent1_list))
    rel1_list = list(set(rel1_list))
    ent_to_no_map = dict()
    for idx, ent in enumerate(ent1_list):
        ent_to_no_map[ent] = idx
    ent2_list = []
    rel2_list = []
    for head, rel, tail in triple2_list:
        ent2_list.extend([head, tail])
        rel2_list.append(rel)
    ent2_list = list(set(ent2_list))
    rel2_list = list(set(rel2_list))
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
    for rel1 in tqdm(rel1_list, desc="count subrelation"):
        ent_pair_set1 = set(rel_to_ent_map1[rel1])
        for rel2 in rel2_list:
            ent_pair_set2 = set(rel_to_ent_map2[rel2])
            interset = ent_pair_set1.intersection(ent_pair_set2)
            subrel_map[("kg1:"+rel1, "kg2:"+rel2)] = len(interset) / len(ent_pair_set1)
            subrel_map[("kg2:" + rel2, "kg1:" + rel1)] = len(interset) / len(ent_pair_set1)
    return subrel_map


def build_graph(triples):
    inbound_map = dict()
    for h, r, t in triples:
        if t not in inbound_map:
            inbound_map[t] = []
        inbound_map[t].append((r,h))
    return inbound_map


def count_consistency(kg1_triples, kg2_triples, alignment, grouding_alignment):
    tail_to_relnheads_map1 = build_graph(kg1_triples)
    tail_to_relnheads_map2 = build_graph(kg2_triples)

    func_map1 = functionality(kg1_triples)
    func_map2 = functionality(kg2_triples)
    subrel_map = subsumption(kg1_triples, kg2_triples, grouding_alignment)

    check_align_map = {pair: True for pair in grouding_alignment}

    ent1_to_consist = dict()
    for ent1, ent2 in tqdm(alignment, desc="compute consistency"):
        relnheads1 = tail_to_relnheads_map1[ent1]
        relnheads2 = tail_to_relnheads_map2[ent2]
        accu_prob = 1
        for rel1, head1 in relnheads1:
            for rel2, head2 in relnheads2:
                rel1_func = func_map1[rel1]
                rel2_func = func_map2[rel2]
                equiv = float(check_align_map.get((head1, head2), False))
                sub_prob = (1 - subrel_map[("kg1:"+rel1, "kg2:"+rel2)] * rel2_func * equiv) * (1 - subrel_map[("kg2:"+rel2, "kg1:"+rel1)] * rel1_func * equiv)
                accu_prob *= sub_prob
        prob = 1 - accu_prob
        ent1_to_consist[ent1] = prob

    return ent1_to_consist



def load_results(res_dir):
    res_fn = os.path.join(res_dir, "alignment_results_12")
    kg1_ent_ids_fn = os.path.join(res_dir, "kg1_ent_ids")
    kg2_ent_ids_fn = os.path.join(res_dir, "kg2_ent_ids")
    res_alignment = read_links(res_fn)
    kg1_ent_ids = read_links(kg1_ent_ids_fn)
    kg2_ent_ids = read_links(kg2_ent_ids_fn)
    kg1_id2ent_map = {id: ent for ent, id in kg1_ent_ids}
    kg2_id2ent_map = {id: ent for ent, id in kg2_ent_ids}
    alignment = [(kg1_id2ent_map[ent1], kg2_id2ent_map[ent2]) for ent1, ent2 in res_alignment]
    return alignment


def main_compute_consistency():
    data_dir = "/home/uqbliu3/experiments/datasets/D_W_15K_V1/"
    res_dir = "/home/uqbliu3/experiments/output/results/AliNet/D_W_15K_V1/721_5fold/1/20210616113552"
    out_dir = "/home/uqbliu3/experiments/output/results/"

    kg1_triples, kg1_entities, kg1_relations = read_relation_triples(os.path.join(data_dir, "rel_triples_1"))
    kg2_triples, kg2_entities, kg2_relations = read_relation_triples(os.path.join(data_dir, "rel_triples_2"))
    kg1_triples = list(kg1_triples)
    kg2_triples = list(kg2_triples)
    grouding_alignment = read_links(os.path.join(data_dir, "ent_links"))
    ent1_to_ent2_map = dict(grouding_alignment)
    res_alignment = load_results(res_dir)
    inv_kg1_triples = [(t,"inv_"+r,h) for h, r, t in kg1_triples]
    inv_kg2_triples = [(t, "inv_" + r, h) for h, r, t in kg2_triples]
    kg1_triples.extend(inv_kg1_triples)
    kg2_triples.extend(inv_kg2_triples)

    res_consis_map = count_consistency(kg1_triples, kg2_triples, res_alignment, grouding_alignment)
    correct_res_consis_map = {ent1: res_consis_map[ent1] for ent1, ent2 in res_alignment if ent2 == ent1_to_ent2_map[ent1]}
    wrong_res_consis_map = {ent1: res_consis_map[ent1] for ent1, ent2 in res_alignment if ent2 != ent1_to_ent2_map[ent1]}


    label_alignment = [(ent1, ent1_to_ent2_map[ent1]) for ent1, ent2 in res_alignment]
    label_consis_map = count_consistency(kg1_triples, kg2_triples, label_alignment, grouding_alignment)

    with open(os.path.join(out_dir, "consistency.json"), "w+") as file:
        obj = {"correct_pred": correct_res_consis_map, "wrong_pred": wrong_res_consis_map, "label": label_consis_map}
        file.write(json.dumps(obj))


def main_plot_consistency(out_dir):

    with open(os.path.join(out_dir, "consistency.json")) as file:
        cont = file.read()
        obj = json.loads(cont)

    df = pd.DataFrame(data={
        "prob": list(obj["label"].values()) + list(obj["correct_pred"].values()) + list(obj["wrong_pred"].values()),
        "category": ["label"]*len(obj["label"]) + ["correct_pred"]*len(obj["correct_pred"]) + ["wrong_pred"]*len(obj["wrong_pred"])
    })
    # sb.displot(df, x="prob", hue="category", multiple="dodge", shrink=.8)
    sb.displot(df, x="prob", hue="category", element="poly")
    plt.savefig(os.path.join(out_dir, "consistency.png"))



def main_check_conflicts():
    res_dir = "/home/uqbliu3/experiments/output/results/AliNet/D_W_15K_V1/721_5fold/1/20210616113552"
    res_alignment = load_results(res_dir)
    ent1_list = [ent2 for ent1, ent2 in res_alignment]

    count = Counter(ent1_list)
    df = pd.DataFrame({"ent2_num": count.values()})
    sb.histplot(df, x="ent2_num", binwidth=1)
    plt.savefig(os.path.join(out_dir, "ent2_num.png"))



if __name__ == "__main__":
    out_dir = "/home/uqbliu3/experiments/output/results/"

    # main_compute_consistency()
    # main_plot_consistency(out_dir)

    main_check_conflicts()









