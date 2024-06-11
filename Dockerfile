# Use a Debian-based Python image
FROM python:3.8-slim

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget && apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

COPY . /app

WORKDIR /app

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Install system level dependencies not available via conda or pip
RUN apt-get update && apt-get install -y swig

RUN conda config --add channels conda-forge

# Create the conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate the Conda environment and install faiss-cpu
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate simple_retrieval_augmented_generation_env"

# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 8505

# Make the script executable
RUN chmod +x /app/start.sh

ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate simple_retrieval_augmented_generation_env &&./start.sh"]
