#!/bin/bash
# Convenience script to activate the SynLLM environment and check GPU status

# Handle different shell setups
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    export PATH="$(conda info --base)/bin:$PATH"
fi

# Activate the environment
conda activate SynLLM

# Display GPU status
echo "Checking NVIDIA GPU status..."
nvidia-smi

# Verify CUDA with PyTorch
echo "Verifying CUDA with PyTorch..."
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Check memory usage
echo "Checking GPU memory usage..."
python -c "import GPUtil; [print(f'GPU {gpu.id}: {gpu.name}, Memory Used: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)') for gpu in GPUtil.getGPUs()]"
