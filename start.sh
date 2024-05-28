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
conda config --set pip_interop_enabled True

conda install -y -c conda-forge faiss-cpu==1.7.3

conda install -y langchain==0.1.20
conda install -y langsmith==0.1.63
conda install -y numpy==1.26.4
conda install -y pandas==2.2.2
conda install -y scikit-learn==1.5.0
conda install -y streamlit==1.35.0
conda install -y -c pytorch pytorch=2.3.0 torchvision torchaudio -c defaults

conda install -y -c conda-forge transformers==4.41.1
conda install -y pip
pip install -U pyChatGPT

#conda install -y pytest==8.2.1
#conda install -y Faker==25.2.0

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
