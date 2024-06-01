#!/bin/bash

# Ensure Conda environment is properly initialized
source /opt/anaconda3/etc/profile.d/conda.sh

# Create and/or activate the Conda environment with Python version 3.9.19
echo "Creating or updating Conda environment with Python 3.9.19..."
conda create --name rag_env python=3.9.19 -y
conda activate rag_env

# Verify the Conda environment
echo "Active Conda environment:"
conda env list
echo "Python path:"
which python

# Print Python version
echo "*********************************************************Python version:"
python --version

# Additional pip updates and installations
echo "Updating pip and installing packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

## Add conda-forge channel
conda config --add channels conda-forge
conda config --set pip_interop_enabled True

# Install required packages
echo "Installing conda packages..."
conda update --all -y
conda install -y -c intel mkl
conda install -y -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
conda install -y langchain==0.1.20
conda install -y langchain-openai==0.0.8
conda install -y langsmith==0.1.63
conda install -y streamlit==1.35.0
conda install -y -c pytorch pytorch==2.2.2 torchvision torchaudio
conda install -y -c conda-forge transformers==4.41.1

conda install -y nltk
conda install -y pickle
python -m nltk.downloader stopwords
python -m nltk.downloader punkt

#conda install -y pytest==8.2.1  <<< testing pkg
#conda install -y Faker==25.2.0  <<< testing pkg


# Set environment variables for MKL
export MKLROOT=$(conda info --base)/envs/rag_env
echo "MKLROOT: $MKLROOT"
#export DYLD_LIBRARY_PATH=$MKLROOT/lib:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$MKLROOT/lib:/usr/local/lib:DYLD_LIBRARY_PATH

echo "DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MKLROOT/lib:/usr/local/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"


# Print Python version
echo "*********************************************************Python version:"
python --version

# Print faiss-cpu version
echo "*********************************************************faiss-cpu version:"
python -c "import faiss; print(faiss.__version__)"

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