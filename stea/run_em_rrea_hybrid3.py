# -*- coding: utf-8 -*-

from ealib.approach.conf import Config
import os
from stea.neural_rrea import RREAModule
from stea.jointdistr_hybrid3 import Hybrid3JointDistri
from stea.emea_framework import EMEAFramework
import argparse


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='F:\\EA\\emea\\formatted_rrea\\zh_en', type=str)
parser.add_argument('--data_name',default='zh_en', type=str)
parser.add_argument('--output_dir',default='F:\\EA\\emea\\out', type=str)
parser.add_argument('--res_dir',default='F:\\EA\\emea\\res',  type=str)
parser.add_argument('--topK', default=10,type=int)
parser.add_argument('--conflict_checker_neigh_num',default=1, type=int)
parser.add_argument('--em_iteration_num', default=10, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--second_device', default="cuda:0", type=str)
parser.add_argument('--tf_device', default="/gpu:0", type=str)
parser.add_argument('--initial_training', default="supervised", type=str)
parser.add_argument('--max_train_epoch', default=3, type=int)
parser.add_argument('--max_continue_epoch', default=3, type=int)
parser.add_argument('--eval_freq', default=10000, type=int)
parser.add_argument('--neu_save_metrics', default=0, type=int)
parser.add_argument('--ruleset', type=str)
parser.add_argument('--py_exe_fn', default=None, type=str)
args = parser.parse_args()




conf = Config()
conf.update(vars(args))
# conf.device = "cuda:1"

if not os.path.exists(conf.output_dir):
    os.makedirs(conf.output_dir)


neural_ea_model = RREAModule(conf)
#joint_distr_model = ParisJointDistri(conf)
joint_distr_model = Hybrid3JointDistri(conf)
#joint_distr_model_inv = ParisJointDistri(conf,inv=True)
joint_distr_model_inv = Hybrid3JointDistri(conf,inv=True)
em_framework = EMEAFramework(conf=conf, jointdistr_model=joint_distr_model, jointdistr_model_inv=joint_distr_model_inv, neural_ea_model=neural_ea_model)
em_framework.run_EM()
em_framework.evaluate()


