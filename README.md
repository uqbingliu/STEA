# EMEA-ext

This repo contains the source code and used data of paper "Guiding Neural Entity Alignment with Compatibility", which is under review of EMNLP 2022.

## Structure of Folders
```shell
emea_code/
  - datasets/
  - emea/
  - OpenEA/
  - scripts/
  - environment.yml
  - README.md
```
After you run EMEA, there will be a `output/` folder which stores the evaluation results.

## Device
Our experiments are run on one GPU server which is configured with 3 NVIDIA GeForce GTX 2080Ti GPUs and Ubuntu 20.04 OS. 
We suggest you use at least two GPUs in case of Out-Of-Memeory Issue.


## Install Conda Environment
`cd` to the project directory first. Then, run the following command to install the major environment packages.
```shell
conda env create -f environment.yml
```

With the installed environment, you can run EMEA for RREA and Dual-AMN, which are SOTA neural EA models.
If you also want to run EMEA for AliNet, IPTransE, which are used to verify the generality of EMEA, please also install the following packages with `pip`:
```shell
pip install igraph
pip install python-Levenshtein
pip install gensim
pip install dataclasses
```

```shell
conda install -c conda-forge graph-tool==2.29
```



## Acknowledgement
We used the source codes of [RREA](https://github.com/MaoXinn/RREA), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [OpenEA](https://github.com/nju-websoft/OpenEA).