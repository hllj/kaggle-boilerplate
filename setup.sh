#!/bin/bash
# Setup script for the Kaggle competition boilerplate

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml
echo "Conda environment 'kaggle-boilerplate' created."

# Activate the environment 
echo "To activate the environment, run:"
echo "conda activate kaggle-boilerplate"

# Install git pre-commit hooks
echo "Setting up pre-commit hooks..."
conda activate kaggle-boilerplate && pre-commit install
echo "Pre-commit hooks installed."

# Create necessary directories if they don't exist
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external
mkdir -p models
mkdir -p submission

# Kaggle API setup instructions
echo ""
echo "IMPORTANT: Kaggle API Setup Instructions"
echo "----------------------------------------"
echo "1. Go to https://www.kaggle.com/<username>/account"
echo "2. Scroll down to 'API' section and click 'Create New API Token'"
echo "3. Move the downloaded 'kaggle.json' to ~/.kaggle/kaggle.json"
echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "Setup complete!" 