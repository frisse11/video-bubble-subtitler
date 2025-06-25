#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Colors for better output ---
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}--- Video Bubble Subtitler Installation Script ---${NC}"
echo "This script will help you set up the Python environment and install necessary dependencies."
echo ""

# --- 1. Check for FFmpeg system-wide installation ---
echo -e "${YELLOW}Checking for FFmpeg...${NC}"
if ! command -v ffmpeg &> /dev/null
then
    echo -e "${RED}FFmpeg is not found.${NC}"
    echo "FFmpeg is required for video and audio processing. Please install it on your system."
    echo "Common installation commands:"
    echo "  - Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg"
    echo "  - Fedora: sudo dnf install ffmpeg"
    echo "  - macOS (with Homebrew): brew install ffmpeg"
    echo "  - Windows (with Chocolatey): choco install ffmpeg"
    echo "After installation, please re-run this script."
    exit 1
else
    echo -e "${GREEN}FFmpeg found!${NC}"
fi
echo ""

# --- 2. Choose environment type ---
ENV_TYPE=""
while true; do
    echo -e "${YELLOW}Which Python environment do you want to use?${NC}"
    echo "  1) Conda (Recommended for GPU support and easier dependency management)"
    echo "  2) Python Virtual Environment (venv)"
    read -p "Enter 1 or 2: " choice
    case $choice in
        1) ENV_TYPE="conda"; break;;
        2) ENV_TYPE="venv"; break;;
        *) echo -e "${RED}Invalid choice. Please enter 1 or 2.${NC}";;
    esac
done
echo ""

# --- 3. Setup the chosen environment ---

if [ "$ENV_TYPE" == "conda" ]; then
    echo -e "${YELLOW}Setting up Conda environment...${NC}"
    if ! command -v conda &> /dev/null
    then
        echo -e "${RED}Conda is not found. Please install Anaconda or Miniconda first.${NC}"
        echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    ENV_NAME="video-bubbles"

    # Check if Conda environment already exists
    if conda env list | grep -q "$ENV_NAME"; then
        echo -e "${YELLOW}Conda environment '$ENV_NAME' already exists. Activating it...${NC}"
        # Make sure to activate it for subsequent pip installs
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
    else
        echo "Creating Conda environment: $ENV_NAME"
        conda create -n "$ENV_NAME" python=3.10 -y
        echo "Activating Conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
    fi

    # Install PyTorch based on user's GPU choice
    GPU_CHOICE=""
    while true; do
        read -p "Do you have an NVIDIA GPU for CUDA support? (y/n): " GPU_CHOICE
        case $GPU_CHOICE in
            [Yy]* ) 
                CUDA_VERSION=""
                while true; do
                    read -p "What CUDA version do you have installed? (e.g., 11.8, 12.1, 12.x for latest CPU fallback if unsure): " CUDA_VERSION
                    if [[ "$CUDA_VERSION" =~ ^(11\.[0-9]|12\.[0-9]|12\.x|cpu)$ ]]; then
                        break
                    else
                        echo -e "${RED}Invalid CUDA version. Please enter a valid version (e.g., 11.8, 12.1, 12.x) or 'cpu'.${NC}"
                    fi
                done
                
                if [ "$CUDA_VERSION" == "cpu" ]; then
                    PYTORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                    echo -e "${YELLOW}Installing CPU version of PyTorch...${NC}"
                else
                    PYTORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"
                    echo -e "${YELLOW}Installing PyTorch with CUDA ${CUDA_VERSION}...${NC}"
                fi
                break;;
            [Nn]* ) 
                PYTORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                echo -e "${YELLOW}Installing CPU version of PyTorch...${NC}"
                break;;
            *) echo -e "${RED}Please answer y or n.${NC}";;
        esac
    done
    
    $PYTORCH_INSTALL_CMD || { echo -e "${RED}Failed to install PyTorch. Please check your internet connection or CUDA setup.${NC}"; exit 1; }
    echo -e "${GREEN}PyTorch installed.${NC}"

    echo "Installing other Python dependencies..."
    pip install -r <(echo -e "gradio\nPillow\ntransformers\nlibrosa\nsoundfile\nnumpy\nsentencepiece\naccelerate")
    echo -e "${GREEN}Other dependencies installed.${NC}"

elif [ "$ENV_TYPE" == "venv" ]; then
    echo -e "${YELLOW}Setting up Python Virtual Environment...${NC}"
    ENV_DIR=".venv"
    # Check if venv already exists (and create if not)
    if [ ! -d "$ENV_DIR" ]; then
        python3 -m venv "$ENV_DIR"
    else
        echo -e "${YELLOW}Virtual environment '$ENV_DIR' already exists.${NC}"
    fi

    echo "Activating virtual environment..."
    source "$ENV_DIR/bin/activate"
    echo -e "${GREEN}Virtual environment activated.${NC}"

    # Install PyTorch (CPU only for simplicity in venv, for CUDA, user needs more specific manual install)
    echo -e "${YELLOW}Installing CPU version of PyTorch (for simplicity in venv)...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || \
    { echo -e "${RED}Failed to install PyTorch. Please check your internet connection.${NC}"; exit 1; }
    echo -e "${GREEN}PyTorch installed.${NC}"

    echo "Installing other Python dependencies..."
    pip install -r <(echo -e "gradio\nPillow\ntransformers\nlibrosa\nsoundfile\nnumpy\nsentencepiece\naccelerate")
    echo -e "${GREEN}Other dependencies installed.${NC}"
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo "To run the application:"

if [ "$ENV_TYPE" == "conda" ]; then
    echo "1. Activate your environment: ${YELLOW}conda activate $ENV_NAME${NC}"
    echo "2. Run the app: ${YELLOW}python app.py${NC}"
elif [ "$ENV_TYPE" == "venv" ]; then
    echo "1. Activate your environment: ${YELLOW}source .venv/bin/activate${NC}"
    echo "2. Run the app: ${YELLOW}python app.py${NC}"
fi

echo ""
echo "When you are done, you can deactivate the environment:"
if [ "$ENV_TYPE" == "conda" ]; then
    echo "  ${YELLOW}conda deactivate${NC}"
elif [ "$ENV_TYPE" == "venv" ]; then
    echo "  ${YELLOW}deactivate${NC}"
fi