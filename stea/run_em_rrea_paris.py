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
parser.add_argument('--data_dir',default='F:\\EA\\emea\\formatted_rrea\\zh_en', type=str)
parser.add_argument('--data_name',default='zh_en', type=str)
parser.add_argument('--output_dir',default='F:\\EA\\emea\\out', type=str)
parser.add_argument('--res_dir',default='F:\\EA\\emea\\res',  type=str)
parser.add_argument('--topK', default=10,type=int)
parser.add_argument('--em_iteration_num',default=10, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--second_device', default="cuda:0", type=str)
parser.add_argument('--tf_device', default="/gpu:0", type=str)
parser.add_argument('--initial_training', default="supervised", type=str)
parser.add_argument('--continue_training', default="supervised", type=str)
parser.add_argument('--max_train_epoch', default=250, type=int)
parser.add_argument('--max_continue_epoch', default=50, type=int)
parser.add_argument('--eval_freq', default=1000, type=int)
parser.add_argument('--neu_save_metrics', default=0, type=int)
parser.add_argument('--py_exe_fn', default=None, type=str)
parser.add_argument('--ea_model', default="rrea", type=str)
parser.add_argument('--openea_arg_fn', type=str, default="", required=False)
parser.add_argument('--openea_script_fn', type=str, default="", required=False)
parser.add_argument('--no_joint_distr', default="False", type=str)
parser.add_argument('--neural_thr', default=0, type=float)
parser.add_argument('--joint_distr_thr', default=0, type=float)
parser.add_argument('--neural_mn', default="False", type=str)
parser.add_argument('--joint_distr_mn', default="False", type=str)
parser.add_argument('--joint_distr_pri', default=True, type=bool)
parser.add_argument('--overlapping_only', default=False, type=bool)
parser.add_argument('--joint_distr_pseudo', default=0, type=float)
parser.add_argument('--neural_one2one', default="False", type=str)
parser.add_argument('--joint_distr_one2one', default="False", type=str)
parser.add_argument('--direction', default="source", type=str)
parser.add_argument('--no_sim2prob', default="False", type=str)



args = parser.parse_args()


conf = Config()
conf.update(vars(args))
#conf.device = "cuda:1"

# if args.ea_model.lower() == "rrea" or args.ea_model.lower() == "dual_amn":
#     conf.py_exe_fn = None

if args.py_exe_fn == "":
    conf.py_exe_fn = None
print("eval freq", conf.eval_freq)
print("eval freq", conf.py_exe_fn)
    
if conf.neural_mn == 'True':
    conf.neural_mn=True
if conf.neural_mn == 'False':
    conf.neural_mn=False
if conf.joint_distr_mn == 'True':
    conf.joint_distr_mn=True
if conf.joint_distr_mn == 'False':
    conf.joint_distr_mn=False


if not os.path.exists(conf.output_dir):
    os.makedirs(conf.output_dir)

if args.ea_model.lower() == "rrea":
    neural_ea_model = RREAModule(conf)
elif args.ea_model.lower() == "dual_amn":
    neural_ea_model = DualAMNModule(conf)
elif args.ea_model.lower() == "gcnalign":
    from neural_gcnalign2 import GCNAlignModule
    neural_ea_model = GCNAlignModule(conf)
else:
    neural_ea_model = OpenEAModule(conf)
if conf.no_joint_distr == "True":
    joint_distr_model = None
    joint_distr_model_inv = None
else:
    joint_distr_model = ParisJointDistri(conf)
    joint_distr_model_inv = ParisJointDistri(conf, inv=True)
em_framework = EMEAFramework(conf=conf, jointdistr_model=joint_distr_model, jointdistr_model_inv=joint_distr_model_inv, neural_ea_model=neural_ea_model)
# em_framework.initialize()
em_framework.run_EM()
em_framework.evaluate()




# seed=1
# conflict_checker_neigh_num=5
# task="emea_${data_name}_${train_percent}"



# cf_alpha1=0.8
# cf_alpha2=1.0