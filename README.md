# Targetdiff_finetune
SARS-COV-2 Mpro inhibitor generation mini project

## Environments
```bash
docker push seokwooyun/targetdiff_finetune:latest
```
```bash
docker run --gpus all -it --name targetdiff pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel /bin/bash

conda install pytorch-sparse -c pyg
conda install pytorch-scatter -c pyg
conda install pytorch-cluster -c pyg
conda install conda-forge::python-lmdb
echo 'export PYTHONPATH=/root/targetdiff:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
pip install easydict
pip install rdkit
pip install torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
conda install pyg -c pyg
conda install conda-forge::tensorboard
conda install conda-forge::openbabel

apt-get update
apt-get install -y git

#vina
pip install meeko==0.1.dev3 scipy==1.13 pdb2pqr
conda install -c conda-forge numpy boost-cpp swig
pip install vina #1.2.5 version
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
apt-get update
apt-get install -y libxrender1

#pre-processing for vina docking
conda install bioconda::pubchempy  # to get 3d coords for ligands
pip install pandas

pip install "numpy<2"
```
