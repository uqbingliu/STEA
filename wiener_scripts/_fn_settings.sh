

# related directories and filenames


dataset_root_dir="${proj_dir}/datasets/"
output_root_dir="${proj_dir}/output/emea/"

data_dir="${dataset_root_dir}/cache/${data_name}/${task}/"
output_dir="${output_root_dir}/${data_name}/${task}"
res_dir="${output_dir}/results/${data_name}/"


if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

if [ ! -d $res_dir ]; then
    mkdir -p $res_dir
fi

if [ ! -d "${dataset_root_dir}/cache/${data_name}/" ]; then
    mkdir -p "${dataset_root_dir}/cache/${data_name}/"
fi

