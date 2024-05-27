#!/bin/bash

# Initialize Conda for bash and zsh shells
echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda init bash" >> ~/.bashrc
echo "conda init zsh" >> ~/.zshrc

# Activate the Conda environment
conda activate myenv

# After activating the Conda environment
echo "Active Conda environment:"
conda env list
echo "Python path:"
which python


# Source the shell profile to apply Conda environment
source ~/.bashrc
source ~/.zshrc

# Install required packages
echo "Install conda packages"
conda install -y -c conda-forge transformers
conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
conda install -y streamlit
conda install -y -c conda-forge faiss-cpu
conda install -y pandas
conda install -y numpy
conda install -y ChatGPT



# Navigate to the directory containing scripts and data
cd /app || { echo "Failed to change directory to /app"; exit 1; }

# Run the preprocessing script
python -m rag_application.modules.preprocess_data || { echo "Preprocessing failed"; exit 1; }

# Run the FAISS vector index script
python -m rag_application.modules.vector_index_faiss || { echo "FAISS vector index build failed"; exit 1; }

# Start Streamlit
exec streamlit run rag_application/modules/user_interface.py --server.port=8505 --server.address=0.0.0.0 || { echo "Streamlit failed to start"; exit 1; }
