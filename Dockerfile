# Use a Debian-based Python image
FROM python:3.10-buster

WORKDIR /app
#
## Install dependencies for downloading and executing Miniconda installer
#RUN apt-get update && \
#    apt-get install -y wget && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*
#
## Download the specific version of the Anaconda installer
#RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
#
## Execute the installer
#RUN bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -u -p /opt/conda
#
## Remove the installer script
#RUN rm Anaconda3-2024.02-1-Linux-x86_64.sh


# Install dependencies for downloading and executing Miniconda installer
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Copy the initialization script into the container
COPY init_conda.sh /app/init_conda.sh

# Make the script executable
RUN chmod +x /app/init_conda.sh

# Execute the script
CMD ["/app/init_conda.sh"]

# Download the specific version of the Anaconda installer
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Install Anaconda
RUN bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -u -p /opt/conda

# Cleanup: remove the installer script
RUN rm Anaconda3-2024.02-1-Linux-x86_64.sh

# Initialize Miniconda for bash and zsh shells
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda init bash" >> ~/.bashrc
RUN echo "conda init zsh" >> ~/.zshrc

## Set the PATH environment variable to include the conda binaries
#ENV PATH="/opt/conda/bin:${PATH}"
#
#
#
## Source the Conda initialization scripts to make 'conda' available in this session
#ENV PATH="~/miniconda3/bin:${PATH}"
#RUN source ~/miniconda3/etc/profile.d/conda.sh

## Create a new Conda environment and activate it
#RUN conda create -n myenv python=3.10
#RUN conda activate myenv

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
COPY requirements.txt /app/requirements.txt
RUN grep -vE '^(pytorch|faiss-cpu)' /app/requirements.txt > /app/requirements_filtered.txt

# Install pip dependencies from the filtered requirements file
# TODO: consider using --no-cache-dir if concerned about cache size growth or conflicts with existing cached packages
RUN pip install -r /app/requirements_filtered.txt

# Expose port
EXPOSE 8505

CMD ["./start.sh"]
