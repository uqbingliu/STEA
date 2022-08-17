


# get and enter project directory
#script_fn="${PWD}/${0}"
#script_fn="${0}"
#script_dir=$(dirname $script_fn)
script_dir=/scratch/itee/${USER}/semiea/wiener_scripts/
#script_dir=/scratch/itee/uqswan33/semiea/wiener_scripts/
proj_dir="${script_dir}/../"
cd $proj_dir

module load anaconda/3.6
# prepare conda env
export PYTHONPATH=$proj_dir
eval "$(conda shell.bash hook)"
conda activate emea


py_exe_full_fn=$CONDA_PREFIX/bin/python




