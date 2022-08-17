# -*- coding: utf-8 -*-

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
args = parser.parse_args()

res_dir = args.out_dir

dir_name_list = os.listdir(res_dir)

filtered_dir_name_list = []
for dir_name in dir_name_list:
    if "iteration" in dir_name:
        filtered_dir_name_list.append(dir_name)

num = len(filtered_dir_name_list)

for dir_name in filtered_dir_name_list:
    if dir_name == f"iteration{num-1}":
        continue

    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "emb.npz")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "emb.npz"))
    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "get_emb.ckpt")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "get_emb.ckpt"))
    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "model.ckpt")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "model.ckpt"))
    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "simi2prob_model.ckpt")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "simi2prob_model.ckpt"))
    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "simi2prob_model_inv.ckpt")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "simi2prob_model_inv.ckpt"))
    if os.path.exists(os.path.join(res_dir, dir_name, "joint", "coordinate_ascent_prob_mtx.npz")):
        os.remove(os.path.join(res_dir, dir_name, "joint", "coordinate_ascent_prob_mtx.npz"))
    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "ent_embeds.npy")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "ent_embeds.npy"))
    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "kg1_ent_embeds_txt")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "kg1_ent_embeds_txt"))
    if os.path.exists(os.path.join(res_dir, dir_name, "neu", "kg2_ent_embeds_txt")):
        os.remove(os.path.join(res_dir, dir_name, "neu", "kg2_ent_embeds_txt"))


