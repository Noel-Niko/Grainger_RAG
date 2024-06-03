#!/bin/bash

# Ensure Conda environment is properly initialized
source /opt/anaconda3/etc/profile.d/conda.sh

PYTHON_VERSION="3.9.19"

conda env remove --name rag_env --yes

echo "Creating or updating Conda environment with Python $PYTHON_VERSION..."
conda create --name rag_env python=$PYTHON_VERSION -y

conda activate rag_env

echo "Active Conda environment:"
conda env list
echo "Python path:"
which python

echo "*********************************************************Python version:"
python --version

conda config --add channels defaults
conda config --add channels conda-forge
conda config --set pip_interop_enabled True
conda config --set always_yes True

# Install required packages
echo "Installing conda packages..."
#conda install -c intel mkl=2021.4
#conda install -c conda-forge mkl-service

# Set up MKL paths
#export MKLROOT="${CONDA_PREFIX}/lib"
#export DYLD_LIBRARY_PATH="$MKLROOT/lib":$DYLD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
#export DYLD_FALLBACK_LIBRARY_PATH="${CONDA_PREFIX}/lib"
#
source ~/.bashrc
source ~/.zshrc
conda install -c pytorch faiss-cpu=1.7.4
#conda install -c pytorch faiss-cpu
#conda install -c conda-forge numpy numpy=1.21.2
conda install -c conda-forge numpy

conda install pandas==2.2.1
conda install scikit-learn==1.5.0
conda install -y langchain==0.1.20
conda install -y langchain-openai==0.0.8
conda install -y langchain-community==0.0.19
conda install -y langsmith
conda install -y streamlit==1.35.0
conda install -y -c pytorch pytorch==2.2.2 torchvision torchaudio
conda install -y -c huggingface transformers
conda install -y -c conda-forge huggingface_hub
conda install -y langdetect
conda install pyyaml
conda install packaging
conda install -y nltk
conda install pyarrow
conda install fastparquet
python -m nltk.downloader stopwords
python -m nltk.downloader punkt



#
##conda install -y -c anaconda numpy==1.26.4 mkl-service
#conda install -y -c pytorch faiss-cpu=1.7.3 mkl
#conda install -y -c intel mkl
#conda install -y -c conda-forge numpy mkl-service
#
#
### Set up MKL paths
##export MKLROOT="${CONDA_PREFIX}/lib"
##export DYLD_LIBRARY_PATH="$MKLROOT/lib":$DYLD_LIBRARY_PATH
##export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
##export DYLD_FALLBACK_LIBRARY_PATH="${CONDA_PREFIX}/lib"
###
##source ~/.bashrc
##source ~/.zshrc
#
#conda install pandas==2.2.1
#conda install scikit-learn==1.5.0
##
##export MKLROOT="${CONDA_PREFIX}/lib"
##export DYLD_LIBRARY_PATH="$MKLROOT/lib":$DYLD_LIBRARY_PATH
##export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
##source ~/.bashrc
##source ~/.zshrc
#
#conda install -y langchain==0.1.20
#conda install -y langchain-openai==0.0.8
#conda install -y langsmith==0.1.63
#conda install -y streamlit==1.35.0
#conda install -y -c pytorch pytorch==2.2.2 torchvision torchaudio
##conda install -y -c conda-forge transformers==4.41.1
#conda install -y -c huggingface transformers
#conda install -y -c conda-forge huggingface_hub
#conda install -y langdetect
#conda install pyyaml
#conda install packaging
##conda install -y dask
##conda install -y -c conda-forge dask nltk
#conda install -y nltk
#python -m nltk.downloader stopwords
#python -m nltk.downloader punkt

#conda install -y pytest==8.2.1  <<< testing pkg
#conda install -y Faker==25.2.0  <<< testing pkg

source ~/.bashrc
source ~/.zshrc


# Print Python version
echo "*********************************************************Python version:"
python --version

# Print faiss-cpu version
echo "*********************************************************faiss-cpu version:"
# Print faiss-cpu version
echo "*********************************************************faiss-cpu version:"
python -c 'import faiss; print(faiss.__version__)'


#TODO: *****ADJUST FOR YOUR DEVICE***** for running locally only:
# Set the project root directory as PYTHONPATH
export PYTHONPATH="/Users/noel_niko/PycharmProjects/grainger_rag:$PYTHONPATH"

# Run the Preprocessing file
echo "********************************************************* Preprocessing Data...:"
python rag_application/modules/preprocess_data.py || { echo "Preprocessing failed"; exit 1; }

# Serialize VectorIndex instance
echo "********************************************************* Serializing VectorIndex instance:"
if python rag_application/modules/serialize_vector_index.py; then
    echo "Serialization completed successfully."

    # Define the range of ports to try
    port_range=(8000 8001 8500 8505 9000)

    for port in "${port_range[@]}"; do
        # Use Python to check if the port is available
        python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('localhost', $port)); s.close(); print('Port $port is available')" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            # Port is available, start Streamlit with this port
            echo "Starting Streamlit Application..."
            exec streamlit run rag_application/modules/user_interface.py --server.port=$port --server.address=0.0.0.0 || { echo "Streamlit failed to start"; exit 1; }
            break
        fi
    done

    echo "No available port found in the specified range."
else
    echo "Serialization failed. Exiting."
    exit 1
fi