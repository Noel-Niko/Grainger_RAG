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

# Copy requirements.txt for pip dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Create the conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy



EXPOSE 8505

# Make the script executable
RUN chmod +x /app/start.sh

ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate simple_retrieval_augmented_generation_env && ./start.sh"]
