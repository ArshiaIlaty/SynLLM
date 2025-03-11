#!/bin/bash
# setup_cuda_env.sh - Setup script for creating a CUDA-compatible Python environment with Conda
# Compatible with NVIDIA A100 GPUs
# Usage: bash setup_cuda_env.sh [env_name] [python_version]

# Default values
ENV_NAME=${1:-"cuda_env"}
PYTHON_VERSION=${2:-"3.10"}
CUDA_VERSION="12.4"  # Based on your nvidia-smi output showing CUDA 12.4

# Terminal colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== A100 CUDA Environment Setup ===${NC}"
echo -e "${YELLOW}Setting up environment:${NC} $ENV_NAME"
echo -e "${YELLOW}Python version:${NC} $PYTHON_VERSION"
echo -e "${YELLOW}CUDA version:${NC} $CUDA_VERSION"

# Check if conda is installed and in PATH
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Conda not found in PATH. Checking common locations...${NC}"
    
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
            echo -e "${GREEN}Found Conda at:${NC} $conda_path"
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

# Initialize conda for bash
echo -e "${YELLOW}Initializing conda...${NC}"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if the environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Do you want to remove it? (y/n)${NC}"
    read -r answer
    if [ "$answer" != "${answer#[Yy]}" ]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n "$ENV_NAME"
    else
        echo "Using existing environment. Skipping creation."
        conda activate "$ENV_NAME"
        ENV_EXISTS=true
    fi
fi

# Create a new environment if it doesn't exist or was removed
if [ "$ENV_EXISTS" != true ]; then
    echo -e "${YELLOW}Creating new Conda environment...${NC}"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    
    echo -e "${YELLOW}Activating environment...${NC}"
    conda activate "$ENV_NAME"
    
    # Install CUDA Toolkit compatible packages
    echo -e "${YELLOW}Installing CUDA-compatible packages...${NC}"
    
    # Update conda and install base packages
    conda install -y -c conda-forge conda-build numpy scipy pandas matplotlib jupyterlab ipywidgets
    
    # Install PyTorch with CUDA support
    echo -e "${YELLOW}Installing PyTorch with CUDA $CUDA_VERSION support...${NC}"
    conda install -y pytorch torchvision torchaudio pytorch-cuda=$(echo $CUDA_VERSION | cut -d. -f1-2) -c pytorch -c nvidia
    
    # Install TensorFlow with GPU support
    echo -e "${YELLOW}Installing TensorFlow with GPU support...${NC}"
    conda install -y -c conda-forge tensorflow-gpu
    
    # Install useful data science and ML libraries
    echo -e "${YELLOW}Installing additional data science libraries...${NC}"
    conda install -y -c conda-forge scikit-learn scikit-image seaborn plotly tqdm
    
    # Install utilities
    pip install --no-cache-dir nvidia-ml-py3 gputil
    
    # Install CUDA specific utilities
    # Adjust numba version based on Python version if needed
    conda install -y -c nvidia cuda-python
    conda install -y -c conda-forge numba cudatoolkit=$(echo $CUDA_VERSION | cut -d. -f1-2)
fi

# Verify installation
echo -e "${BLUE}=== Verifying Installation ===${NC}"

echo -e "${YELLOW}Checking CUDA availability with PyTorch:${NC}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device()); print('Device name:', torch.cuda.get_device_name(0))"

echo -e "${YELLOW}Checking TensorFlow GPU support:${NC}"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU')); print('GPU device name:', tf.test.gpu_device_name())"

echo -e "${YELLOW}Checking CUDA with Numba:${NC}"
python -c "from numba import cuda; print('CUDA available:', cuda.is_available()); print('CUDA devices:', cuda.list_devices())"

echo -e "${GREEN}=== Environment Setup Complete ===${NC}"
echo -e "${GREEN}To activate this environment, use:${NC}"
echo -e "conda activate $ENV_NAME"

# Create a convenience activation script
ACTIVATE_SCRIPT="activate_$ENV_NAME.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Convenience script to activate the $ENV_NAME environment and check GPU status

# Activate the environment
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Display GPU status
echo "Checking NVIDIA GPU status..."
nvidia-smi

# Verify CUDA access from Python
echo "Verifying CUDA with PyTorch..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
EOF

chmod +x "$ACTIVATE_SCRIPT"
echo -e "${GREEN}Created activation script:${NC} $ACTIVATE_SCRIPT"

# Optional: Check if Jupyter kernel needs to be installed
echo -e "${YELLOW}Do you want to install a Jupyter kernel for this environment? (y/n)${NC}"
read -r answer
if [ "$answer" != "${answer#[Yy]}" ]; then
    conda install -y -c conda-forge ipykernel
    python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python $PYTHON_VERSION ($ENV_NAME)"
    echo -e "${GREEN}Jupyter kernel installed.${NC}"
fi

# Display final message
echo -e "${BLUE}=== Setup Complete ===${NC}"
echo -e "${GREEN}Your CUDA-enabled Python environment is ready!${NC}"
