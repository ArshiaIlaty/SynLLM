#!/bin/bash

# Create and activate conda environment
conda create -y -n synllm python=3.10
conda activate synllm

# Install required packages
pip install pandas numpy scipy scikit-learn torch transformers psutil

# Create necessary directories if they don't exist
mkdir -p reports
mkdir -p bash

# Run the solo-prompt.py script with a prompt file
python solo-prompt.py --model_name "mistralai/Mistral-7B-Instruct-v0.2" --prompt_file "prompts/prompt1.txt"
