# STEA

This repo contains the source code of paper "Dependency-aware Self-training for Entity Alignment", which has been accepted at WSDM 2023.

Download the used data from this [Dropbox directory](https://www.dropbox.com/sh/8agq4ta2sjtpdhn/AADnIPM-OQKxe6NtF-G9tG3Ua?dl=0).
Decompress it and put it under `STEA_code/` as shown in the folder structure below.

:pushpin: The code has been tested. Feel free to create [issues](https://github.com/uqbingliu/STEA/issues) if you cannot run it successfully. Thanks!

## Structure of Folders
```shell
STEA_code/
  |- datasets/
  |- OpenEA/
  |- scripts/
  |- stea/
    |- Dual_AMN/
    |- GCN-Align/
    |- RREA/
  |- environment.yml
  |- README.md
```
After you run a certain script, the program will automatically create one folder `output/` which stores the evaluation results.

## Device
The configurations of my devices are as below:
* The experiments on 15K datasets were run on one GPU server, which is configured with an Intel(R) Xeon(R) Gold 6128 3.40GHz CPU, 128GB memory, 3 NVIDIA GeForce GTX 2080Ti GPUs and Ubuntu 20.04 OS.
* The experiments on 100K datasets were run on one computing cluster, which runs CentOS 7.8.2003, and allocates us 200GB memory and 2 NVidia Volta V100 SXM2 GPUs.

I think one basic configuration can be: 12GB GPU for 15K datasets, and 32GB GPU for 100K datasets.


## Install Conda Environment
`cd` to the project directory first. Then, run the following command to install the major environment packages.
```shell
conda env create -f environment.yml
```

Activate the env via `conda activate stea`, and then install package `graph-tool`:
```shell
conda install -c conda-forge graph-tool==2.29
```
(It seems slow to install this package. So be patient.)

With the installed environment above, you can run STEA for Dual-AMN, RREA and GCN-Align.

If you also want to run STEA for AliNet, please also install the following packages with `pip`:
```shell
pip install igraph
pip install python-Levenshtein
pip install dataclasses
```

## Run Scripts
Some shell scripts with parameter settings are provided under `scripts/` folder. Some brief
* `run_{Self-training_method}_w_{EA_Model}.sh`. Run a certain self-training method with a certain EA model. You can set the name of dataset, the annotation amount, and other settings as you need.
* `run_analyze_paramK.sh`. Analyze the sensitivity to the hyperparameter `K`.
* `run_analyze_norm_minmax.sh`. Replace the softmax-based normalisation module with a MinMax scaler for analyzing the necessity of our normalisation module.

For each task, the evaluation results as well as some other outputs can be found in a certain folder under the `output/` directory.

Note: AliNet runs much slower than the other EA models. So you can explore the self-training methods with the other EA models first.

## You Want to Report Issues?
We are willing to hear from you
if you have any problem in running our code, or find inconsistency between your running results and what reported in the paper.

## Acknowledgement
We used the source codes of [RREA](https://github.com/MaoXinn/RREA), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [OpenEA](https://github.com/nju-websoft/OpenEA), and [GCN-Align](https://github.com/1049451037/GCN-Align).
