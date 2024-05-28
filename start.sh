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
conda config --env --set default_python 3.10

# Add conda-forge channel
conda config --add channels conda-forge

# Install required packages
echo "Install conda packages"
conda install -y pip
pip install -U pyChatGPT
conda install -y -c conda-forge transformers
conda install -y -c pytorch pytorch=2.3.0 torchvision torchaudio -c pytorch -c defaults cpuonly
conda install -y streamlit
conda install -y -c conda-forge faiss-cpu=1.8.0
conda install -y pandas
conda install -y numpy

# Navigate to the directory containing scripts and data
cd /app || { echo "Failed to change directory to /app"; exit 1; }

echo "*********************************************************Python version:"
python --version

# Print faiss-cpu version
echo "*********************************************************faiss-cpu version:"
pip show faiss-cpu | grep Version

echo "*********************************************************fconfig --show channels:"
conda config --show channels

# Run the preprocessing script
python rag_application/modules/preprocess_data.py || { echo "Preprocessing failed"; exit 1; }

# Run the FAISS vector index script
python rag_application/modules.vector_index_faiss.py || { echo "FAISS vector index build failed"; exit 1; }

# Start Streamlit
exec streamlit run rag_application/modules/user_interface.py --server.port=8505 --server.address=0.0.0.0 || { echo "Streamlit failed to start"; exit 1; }
