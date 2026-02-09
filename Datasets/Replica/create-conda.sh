#!/bin/bash
# =============================================================================
# create-conda.sh
# Sets up a Python 3.9 conda environment for Replica / Habitat-Sim
# Usage: source ./create-conda.sh
# =============================================================================

ENV_NAME="replica"
PYTHON_VERSION="3.9"
HABITAT_VERSION="0.3.3"

echo "=== Checking Conda initialization ==="
# Initialize conda if not already initialized
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Running 'conda init'."
    # Try to find conda in default location
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: Conda not installed or not found."
        return 1
    fi
fi

# Make sure conda is initialized for bash
eval "$(conda shell.bash hook)"

echo "=== Creating Conda environment: $ENV_NAME ==="
# Create environment if it doesn't exist
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

echo "=== Activating environment: $ENV_NAME ==="
conda activate "$ENV_NAME"

echo "=== Installing Habitat-Sim $HABITAT_VERSION ==="
# Install Habitat-Sim
conda install -y habitat-sim="$HABITAT_VERSION=py3.9_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24" \
    -c aihabitat -c conda-forge

echo "=== Installing open3d ==="
conda install -c conda-forge open3d

echo "=== Setup complete ==="
echo "To activate this environment in a new terminal, run:"
echo "    conda activate $ENV_NAME"
