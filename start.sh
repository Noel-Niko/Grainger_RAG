#!/bin/bash

# Source the shell profile to apply Conda environment
source ~/.bashrc
source ~/.zshrc


# After activating the Conda environment
echo "Active Conda environment:"
conda env list
echo "Python path:"
which python

# Activate the Conda environment
conda activate myenv


# Install required packages
echo "Install conda packages"
conda install -y pip
pip install -U pyChatGPT
conda install -y -c conda-forge transformers
conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
conda install -y streamlit
conda install -y -c conda-forge faiss-cpu
conda install -y pandas
conda install -y numpy

# Navigate to the directory containing scripts and data
cd /app || { echo "Failed to change directory to /app"; exit 1; }

# Run the preprocessing script
python /modules.preprocess_data || { echo "Preprocessing failed"; exit 1; }

# Run the FAISS vector index script
python /modules.vector_index_faiss || { echo "FAISS vector index build failed"; exit 1; }

# Start Streamlit
exec streamlit run /modules/user_interface.py --server.port=8505 --server.address=0.0.0.0 || { echo "Streamlit failed to start"; exit 1; }
