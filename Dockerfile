## Use a Debian-based Python image
#FROM python:3.10-buster
#SHELL ["/bin/bash", "-c"]
#
#WORKDIR /app
#
## Install dependencies for downloading and executing Miniconda installer
#RUN apt-get update && \
#    apt-get install -y wget && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*
#
## Download the Anaconda installer
#RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
#
## Install Anaconda
#RUN bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -u -p /opt/conda
#
## Cleanup: remove the installer script
#RUN rm Anaconda3-2024.02-1-Linux-x86_64.sh
#
## Install Rust compiler and Cargo and export the PATH variable
#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default && \
#    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> $HOME/.bashrc && \
#    echo 'source "$HOME/.cargo/env"' >> $HOME/.bashrc && \
#    source $HOME/.bashrc && \
#    rustc --version && \
#    cargo --version
#
## Update package lists and install protobuf development files and compiler
#RUN apt-get update && \
#    apt-get install -y libprotobuf-dev protobuf-compiler && \
#    apt-get -y install vim
#
## Make the script executable
#RUN #chmod +x /app/init_conda.sh
#
## Copy over source code
#COPY . /app
## TODO: UPDATE PERMISSIONS TO 755 FOR PROD
#RUN chmod -R 755 /app
#RUN mkdir -p /rag_application/model/huggingface/hub
#
#RUN pip install --upgrade pip
#
## Copy requirements.txt and filter out conda-installed packages
#COPY requirements.txt /app/requirements.txt
#RUN grep -vE '^(pytorch|faiss-cpu|transformers|streamlit)' /app/requirements.txt > /app/requirements_filtered.txt
#
## Install pip dependencies from the filtered requirements file
## TODO: consider using --no-cache-dir if concerned about cache size growth or conflicts with existing cached packages
#RUN pip install -r /app/requirements_filtered.txt
#
## Example modification in your Dockerfile or startup script
#ENV PYTHONPATH="/opt/conda/envs/myenv/lib/python3.10/site-packages"
#
## Expose port
#EXPOSE 8505
### Make the script executable
##RUN #chmod +x /app/start.sh
## Create a Conda environment named 'myenv'
#RUN conda create --name myenv python=3.10
#
#CMD ["./start.sh"]

# Use the Miniconda3 image as the base, which is based on Python 3
FROM continuumio/miniconda3

# Specify the shell to use for running commands
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# TODO: UPDATE PERMISSIONS TO 755 FOR PROD
RUN chmod -R 755 /app
RUN mkdir -p /rag_application/model/huggingface/hub

# Install curl
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \

# Install Rust compiler and Cargo and export the PATH variable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> $HOME/.bashrc && \
    echo 'source "$HOME/.bashrc"' >> $HOME/.bash_profile && \
    source $HOME/.bash_profile && \
    rustc --version && \
    cargo --version || (echo "Rust installation failed"; false)

# Update package lists and install protobuf development files and compiler
RUN apt-get update && \
    apt-get install -y libprotobuf-dev protobuf-compiler && \
    apt-get -y install vim


RUN pip install --upgrade pip

# Copy over source code
COPY . /app

RUN conda install python=3.10

# Install Conda dependencies
RUN conda install -y -c conda-forge transformers \
        -c pytorch pytorch torchvision torchaudio cpuonly \
        streamlit \
        -c conda-forge faiss-cpu \
        pandas \
        numpy \
        ChatGPT

# Copy requirements.txt and filter out conda-installed packages
COPY requirements.txt /app/requirements.txt
RUN grep -vE '^(pytorch|faiss-cpu|transformers|streamlit)' /app/requirements.txt > /app/requirements_filtered.txt

# Install pip dependencies from the filtered requirements file
# TODO: consider using --no-cache-dir if concerned about cache size growth or conflicts with existing cached packages
RUN pip install -r /app/requirements_filtered.txt

RUN conda create --name myenv python=3.10

# Set PYTHONPATH after creating the Conda environment
ENV PYTHONPATH="/opt/conda/envs/myenv/lib/python3.10/site-packages"

EXPOSE 8505

RUN #chmod +x /app/start.sh

CMD ["./start.sh"]


