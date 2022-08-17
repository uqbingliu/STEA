#!/bin/bash
#SBATCH --job-name=one2one_alinet
#SBATCH --output=wiener_logs/one2one_alinet.out
#SBATCH --error=wiener_logs/one2one_alinet.err
#SBATCH --mail-user=liubing.csai@gmail.com
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=50g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

#script_dir=$(dirname "$PWD/${0}")
script_dir=/scratch/itee/uqbliu3/semiea/wiener_scripts/
. $script_dir/_env_settings.sh

# special task settings
export CUDA_VISIBLE_DEVICES="0,1"
#export CUDA_VISIBLE_DEVICES="0"
tf_device="1"
device="cuda:0"
second_device="cuda:0"
# sup, semi
initial_training=sup
continue_training=sup
seed=0
max_train_epoch=250
max_continue_epoch=50
topK=10
em_iteration_num=10
no_joint_distr=True
neural_one2one=True
neural_thr=0
joint_distr_thr=0
neural_mn=False
joint_distr_mn=True
# source target both
direction=source

joint_distr_thr=0.1

openea_script_fn="${proj_dir}/OpenEA/run/main_from_args.py"
openea_arg_fn="${proj_dir}/OpenEA/run/args/alinet_args_15K.json"


for data_name in "zh_en" "ja_en" "fr_en"
#for data_name in "zh_en"
do
  for train_percent in 0.01 0.05 0.1 0.2 0.3
#  for train_percent in 0.01
  do
    task="overall_perf_alinet_one2one_${initial_training}_${train_percent}_seed${seed}"

    . $script_dir/_fn_settings.sh

    # task cmds
    if [ ! -d ${data_dir} ]; then
        cp -r ${dataset_root_dir}/original_datasets/${data_name}/ ${data_dir}
    fi


    params="--data_dir=${data_dir} --data_name=${data_name} --train_percent=${train_percent} --for_openea"
    echo $params
    python ${proj_dir}/emea/run_prepare_data.py ${params}


    params="--data_dir=${data_dir} --data_name=${data_name} --output_dir=${output_dir} --res_dir=${res_dir}
    --topK=${topK} --em_iteration_num=${em_iteration_num} --initial_training=${initial_training} --continue_training=${continue_training}
    --tf_device=${tf_device} --device=${device} --second_device=${second_device} --py_exe_fn=${py_exe_full_fn}
    --neural_thr=${neural_thr} --joint_distr_thr=${joint_distr_thr} --neural_mn=${neural_mn} --joint_distr_mn=${joint_distr_mn}
    --direction=${direction} --neural_one2one=${neural_one2one} --no_joint_distr=${no_joint_distr}
    --openea_script_fn=${openea_script_fn} --openea_arg_fn=${openea_arg_fn} --ea_model=alinet"
    echo $params
    python ${proj_dir}/emea/run_em_rrea_paris.py ${params}

    python ${proj_dir}/emea/rm_emb_files.py --out_dir=${output_dir}
  done
done



