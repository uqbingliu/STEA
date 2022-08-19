


# get and enter project directory
script_fn="${PWD}/${0}"
#script_fn="${0}"
script_dir=$(dirname $script_fn)
proj_dir="${script_dir}/../"
cd $proj_dir


# prepare conda env
export PYTHONPATH=$proj_dir
eval "$(conda shell.bash hook)"
conda activate stea


py_exe_full_fn=$CONDA_PREFIX/bin/python




