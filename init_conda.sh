#!/bin/bash
# Initialize Conda for bash and zsh shells
echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda init bash" >> ~/.bashrc
echo "conda init zsh" >> ~/.zshrc

# Activate the Conda environment
conda activate myenv
