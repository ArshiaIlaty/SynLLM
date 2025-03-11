#!/bin/bash
# Combined setup script for creating a CUDA-compatible Python environment
# Compatible with NVIDIA A100 GPUs
# Usage: bash setup_cuda_env.sh [env_name] [python_version]

# Default values
ENV_NAME=${1:-"cuda_env"}
PYTHON_VERSION=${2:-"3.10"}

echo "=== A100 CUDA Environment Setup ==="
echo "Setting up environment: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"

# Check if conda is installed and in PATH
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Checking common locations..."
    
    # Check common Conda install locations
    CONDA_PATHS=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "/opt/conda/bin/conda"
        "/usr/local/anaconda3/bin/conda"
        "/usr/local/miniconda3/bin/conda"
    )
    
    for conda_path in "${CONDA_PATHS[@]}"; do
        if [ -f "$conda_path" ]; then
            echo "Found Conda at: $conda_path"
            # Add conda to path for this session
            export PATH="$(dirname "$conda_path"):$PATH"
            break
        fi
    done
    
    # If still not found, exit with error
    if ! command -v conda &> /dev/null; then
        echo "Error: Conda not found. Please install Conda first."
        exit 1
    fi
fi

# Initialize conda for this session
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if the environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Do you want to remove it? (y/n)"
    read -r answer
    if [ "$answer" != "${answer#[Yy]}" ]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME"
    else
        echo "Using existing environment. Skipping creation."
        conda activate "$ENV_NAME"
        ENV_EXISTS=true
    fi
fi

# Create a new environment if it doesn't exist or was removed
if [ "$ENV_EXISTS" != true ]; then
    echo "Creating new Conda environment..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    
    echo "Activating environment..."
    conda activate "$ENV_NAME"
    
    echo "Installing base packages..."
    # Install base scientific and visualization packages
    conda install -y -c conda-forge conda-build numpy scipy pandas matplotlib jupyterlab ipywidgets
    
    echo "Installing PyTorch with CUDA support..."
    # Use pip instead of conda for more reliable CUDA integration
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    echo "Installing Hugging Face Transformers..."
    pip install transformers accelerate
    
    echo "Installing GPU utilities..."
    pip install gputil nvidia-ml-py3

    # Install tf-keras for compatibility with Transformers
    echo "Installing tf-keras for Transformers compatibility..."
    conda install -y -c conda-forge tf-keras
    
    echo "Installing TensorFlow with GPU support..."
    conda install -y -c conda-forge tensorflow-gpu
    
    echo "Installing additional data science libraries..."
    conda install -y -c conda-forge scikit-learn scikit-image seaborn plotly tqdm psutil datasets kaggle
    
    echo "Installing CUDA Python tools..."
    conda install -y -c nvidia cuda-python
    pip install numba
fi

# Verify installation
echo "=== Verifying Installation ==="

echo "Checking PyTorch installation:"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'None')" || echo "PyTorch check failed, but continuing setup."

echo "Checking TensorFlow GPU support:"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU')); print('GPU device name:', tf.test.gpu_device_name() if len(tf.config.list_physical_devices('GPU')) > 0 else 'None')" || echo "TensorFlow check failed, but continuing setup."

echo "Checking Transformers installation:"
python -c "from transformers import pipeline; print('Transformers installation successful')" || echo "Transformers check failed, but continuing setup."

echo "Checking GPUtil:"
python -c "import GPUtil; gpus = GPUtil.getGPUs(); print(f'GPUtil detected {len(gpus)} GPUs')" || echo "GPUtil check failed, but continuing setup."

echo "Checking Numba CUDA:"
python -c "import numba; from numba import cuda; print('CUDA available:', cuda.is_available()); print('CUDA devices:', [device.name for device in cuda.list_devices()] if cuda.is_available() else 'None')" || echo "Numba check failed, but continuing setup."

echo "=== Environment Setup Complete ==="
echo "To activate this environment, use:"
echo "conda activate $ENV_NAME"

# Create a convenience activation script
ACTIVATE_SCRIPT="activate_$ENV_NAME.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Convenience script to activate the $ENV_NAME environment and check GPU status

# Handle different shell setups
if [ -f "\$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "\$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    export PATH="\$(conda info --base)/bin:\$PATH"
fi

# Activate the environment
conda activate $ENV_NAME

# Display GPU status
echo "Checking NVIDIA GPU status..."
nvidia-smi

# Verify CUDA with PyTorch
echo "Verifying CUDA with PyTorch..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Verify imports for your specific packages
echo "Verifying imports for your package list..."
python -c "import gc, json, os, re, time, datetime, GPUtil, pandas, psutil, torch, tqdm; from transformers import pipeline, set_seed; print('All packages imported successfully')"
EOF

chmod +x "$ACTIVATE_SCRIPT"
echo "Created activation script: $ACTIVATE_SCRIPT"

# Jupyter kernel installation
echo "Do you want to install a Jupyter kernel for this environment? (y/n)"
read -r answer
if [ "$answer" != "${answer#[Yy]}" ]; then
    conda install -y -c conda-forge ipykernel
    python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python $PYTHON_VERSION ($ENV_NAME)"
    echo "Jupyter kernel installed."
fi

# Display final message
echo "=== Setup Complete ==="
echo "Your CUDA-enabled Python environment with required packages is ready!"
