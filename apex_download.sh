#!/bin//bash
# Script: apex_download.sh
# purpose: Reproducible installation of NVIDIA Apex for FourCastNet
# Author: garciac2 


# config
PROJECT_DIR="$(pwd)"
ENV_PATH="./env"
APEX_BUILD_DIR="$PROJECT_DIR/../apex"

echo "--- Starting Apex Installation ---"

source "$ENV_PATH/bin/activate"
echo "Environment activated."

cd "$APEX_BUILD_DIR"


# Skip version check
export SKIP_CUDA_CHECK=1
# Support both A30 (8.0) and L40S (8.9)
export TORCH_CUDA_ARCH_LIST="8.0;8.9"

echo "Building Apex with CUDA and C++ extensions..."
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# clean up 
# cd - > /dev/null
# echo "Cleaning up build directory..."
# rm -rf "$APEX_BUILD_DIR"

echo "--- Apex Installation Complete ---"
