# STEA

This repo contains the source code and used data of paper "Dependency-aware Self-training for Entity Alignment", which is under review of WSDM 2022.

## Structure of Folders
```shell
STEA_code/
  - datasets/
  - stea/
  - OpenEA/
  - scripts/
  - environment.yml
  - README.md
```
After you run EMEA, there will be a `output/` folder which stores the evaluation results.

## Device
* The experiments on 15K datasets were run on one GPU server, which is configured with an Intel(R) Xeon(R) Gold 6128 3.40GHz CPU, 128GB memory, 3 NVIDIA GeForce GTX 2080Ti GPUs and Ubuntu 20.04 OS.
* The experiments on 100K datasets were run on one computing cluster, which runs CentOS 7.8.2003, and allocates us 200GB memory and 2 NVidia Volta V100 SXM2 GPUs.

## Install Conda Environment
`cd` to the project directory first. Then, run the following command to install the major environment packages.
```shell
conda env create -f environment.yml
```

Activate the env via `conda activate stea`, and then install package `graph-tool`:
```shell
conda install -c conda-forge graph-tool==2.29
```

With the installed environment, you can run STEA for Dual-AMN, RREA and GCN-Align.
If you also want to run STEA for AliNet, please also install the following packages with `pip`:
```shell
pip install igraph
pip install python-Levenshtein
pip install gensim
pip install dataclasses
```



## Acknowledgement
We used the source codes of [RREA](https://github.com/MaoXinn/RREA), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [OpenEA](https://github.com/nju-websoft/OpenEA), and [GCN-Align](https://github.com/1049451037/GCN-Align).