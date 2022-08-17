# -*- coding: utf-8 -*-

from stea.conf import Config
from stea.emea_framework import EMEAFramework
import os
from stea.jointdistr_paris import ParisJointDistri
import argparse
from stea.neural_rrea import RREAModule
from stea.neural_dual_amn import DualAMNModule
from stea.neural_openea import OpenEAModule

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--data_name', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--res_dir', type=str)
parser.add_argument('--topK', type=int)
parser.add_argument('--em_iteration_num', type=int)
parser.add_argument('--tf_device', default="0", type=str)
parser.add_argument('--initial_training', default="supervised", type=str)
parser.add_argument('--max_train_epoch', default=500, type=int)
parser.add_argument('--max_continue_epoch', default=100, type=int)
parser.add_argument('--neu_save_metrics', default=0, type=int)
parser.add_argument('--py_exe_fn', default=None, type=str)
parser.add_argument('--ea_model', default="rrea", type=str)
parser.add_argument('--openea_arg_fn', type=str, default="", required=False)
parser.add_argument('--openea_script_fn', type=str, default="", required=False)
args = parser.parse_args()


conf = Config()
conf.update(vars(args))
# conf.device = "cuda:1"

if not os.path.exists(conf.output_dir):
    os.makedirs(conf.output_dir)

if args.ea_model.lower() == "rrea":
    neural_ea_model = RREAModule(conf)
elif args.ea_model.lower() == "dual_amn":
    neural_ea_model = DualAMNModule(conf)
else:
    neural_ea_model = OpenEAModule(conf)

joint_distr_model = ParisJointDistri(conf)
# em_framework = EMEAFramework(conf=conf, jointdistr_model=joint_distr_model, neural_ea_model=neural_ea_model)
joint_distr_model_inv = ParisJointDistri(conf, inv=True)
em_framework = EMEAFramework(conf=conf, jointdistr_model=joint_distr_model, jointdistr_model_inv=joint_distr_model_inv, neural_ea_model=neural_ea_model)
em_framework.run_EM()
em_framework.evaluate()



