#!/bin/bash

script_dir=$(dirname "$PWD/${0}")
. $script_dir/_env_settings.sh

# special task settings
export CUDA_VISIBLE_DEVICES="0"
tf_device=""
initial_training=sup
seed=0
max_train_epoch=500
max_continue_epoch=100
em_iteration_num=5
topK=10

#for data_name in "dbp_yg" "dbp_wd"
for data_name in "dbp_yg"
do
#  for train_percent in 0.01 0.05 0.1 0.2 0.3
  for train_percent in 0.01
  do
    task="overall_perf_rrea_${initial_training}_${train_percent}_seed${seed}"

    . $script_dir/_fn_settings.sh

    # task cmds
#    if [ ! -d ${dataset_root_dir}/${data_name}/ ]; then
#        mkdir -p ${dataset_root_dir}/${data_name}/
#    fi
    if [ ! -d ${data_dir} ]; then
        cp -r ${dataset_root_dir}/original_datasets/${data_name}/ ${data_dir}
    fi


    params="--data_dir=${data_dir} --data_name=${data_name} --train_percent=${train_percent}"
    echo $params
    python ${proj_dir}/emea/run_prepare_data.py ${params}


    params="--data_dir=${data_dir} --data_name=${data_name} --output_dir=${output_dir} --res_dir=${res_dir}
    --topK=${topK} --em_iteration_num=${em_iteration_num}
    --tf_device=${tf_device} --initial_training=${initial_training} --max_train_epoch=${max_train_epoch}
    --max_continue_epoch=${max_continue_epoch} --ea_model=rrea
    --py_exe_fn=${py_exe_full_fn}"
    echo $params
    python ${proj_dir}/emea/run_emea.py ${params}

    python ${proj_dir}/emea/rm_emb_files.py --out_dir=${output_dir}
  done
done




