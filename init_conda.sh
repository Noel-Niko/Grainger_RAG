##!/bin/bash
#
## Initialize Conda for bash and zsh shells
#echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
#echo "conda init bash" >> ~/.bashrc
#echo "conda init zsh" >> ~/.zshrc
#
## Activate the Conda environment
#conda activate myenv
#
## After activating the Conda environment
#echo "Active Conda environment:"
#conda env list
#echo "Python path:"
#which python
#
#
## Source the shell profile to apply Conda environment
#source ~/.bashrc
#source ~/.zshrc
#
## Install required packages
#echo "install -y -c conda-forge transformers"
#conda install -y -c conda-forge transformers
#echo "install -y -c pytorch pytorch torchvision torchaudio cpuonly"
#conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
#echo "install -y streamlit"
#conda install -y streamlit
#echo "install -y -c conda-forge faiss-cpu"
#conda install -y -c conda-forge faiss-cpu
