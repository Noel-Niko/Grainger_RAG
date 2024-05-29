# Use a Debian-based Python image
FROM python:3.9

SHELL ["/bin/bash", "-c"]

WORKDIR /app

# Install dependencies for downloading and executing conda installer
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download the specific version of the Anaconda installer
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Install Anaconda
RUN bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -u -p /opt/conda

# Cleanup: remove the installer script
RUN rm Anaconda3-2024.02-1-Linux-x86_64.sh

# Initialize Conda for bash and zsh shells, modify PATH, and create a Conda environment named 'myenv' in a single RUN instruction
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda init bash && conda init zsh && export PATH=/opt/conda/bin:$PATH && conda create -n ragEnv python=3.9"


# Create a symbolic link to Python 3.9
RUN ln -s /opt/conda/envs/ragEnv/bin/python3.9 /opt/conda/bin/python39

# Install Rust compiler and Cargo and export the PATH variable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> $HOME/.bashrc && \
    echo 'source "$HOME/.cargo/env"' >> $HOME/.bashrc && \
    source $HOME/.bashrc && \
    rustc --version && \
    cargo --version

# Update package lists and install protobuf development files and compiler
RUN apt-get update && \
    apt-get install -y libprotobuf-dev protobuf-compiler && \
    apt-get -y install vim

# Copy over source code
COPY . /app
# TODO: UPDATE PERMISSIONS TO 755 FOR PROD
RUN chmod -R 755 /app
RUN mkdir -p /rag_application/model/huggingface/hub

RUN pip install --upgrade pip

# Copy requirements.txt and filter out conda-installed packages
#COPY requirements.txt /app/requirements.txt
#RUN grep -vE '^(pytorch|faiss-cpu|transformers|streamlit)' /app/requirements.txt > /app/requirements_filtered.txt

# Install pip dependencies from the requirements file
# TODO: consider using --no-cache-dir if concerned about cache size growth or conflicts with existing cached packages
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="/opt/conda/envs/ragEnv/lib/python3.9/site-packages"

# Expose port
EXPOSE 8505
## Make the script executable
RUN #chmod +x /app/start.sh

#CMD ["./start.sh"]

CMD ["sh", "-c", "./start.sh"]
