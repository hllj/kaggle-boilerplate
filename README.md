# PyTorch Kaggle Competition Boilerplate

A comprehensive boilerplate for participating in Kaggle competitions using PyTorch. This repository includes structured code for data processing, model training with multiple frameworks, experiment tracking, and automated submission.

## Features

- **Multiple Training Frameworks**: Support for vanilla PyTorch, PyTorch Lightning, and Hugging Face Transformers
- **Experiment Tracking**: Integration with Weights & Biases (WandB) 
- **Configuration Management**: Centralized configuration using Hydra/OmegaConf
- **Evaluation**: Comprehensive metrics calculation and cross-validation
- **Reproducibility**: Seed setting and version control
- **Automation**: GitHub Actions workflow for automated Kaggle submissions

## Project Structure

```
├── .github/workflows/      # GitHub Actions workflows for CI/CD and Kaggle submissions
├── configs/                # Configuration files using Hydra/OmegaConf
├── logs/                   # Training and evaluation logs
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model definitions and training scripts
│   └── evaluation/         # Evaluation metrics and utilities
├── submission/             # Code for generating Kaggle submissions
├── tests/                  # Unit tests
├── environment.yml         # Conda environment specification
└── requirements.txt        # Python dependencies
```

## Getting Started

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pytorch-kaggle-boilerplate.git
   cd pytorch-kaggle-boilerplate
   ```

2. Set up the environment:
   ```bash
   # Using the setup script
   bash setup.sh
   
   # Or manually
   conda env create -f environment.yml
   conda activate kaggle-env
   pip install -e .
   pre-commit install
   ```

3. Configure Kaggle API:
   - Go to Kaggle > Account > Create API Token
   - Place the downloaded `kaggle.json` in `~/.kaggle/` 
   - Set appropriate permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Usage

1. **Data Preparation**:
   ```bash
   # Download competition data
   kaggle competitions download -c [COMPETITION_NAME]
   ```

2. **Exploratory Data Analysis**:
   - Use notebooks in the `notebooks/` directory for EDA

3. **Training**:
   ```bash
   # Vanilla PyTorch
   python src/models/train.py --config configs/default.yaml
   
   # PyTorch Lightning
   python src/models/train_pl.py --config configs/default.yaml
   
   # Hugging Face Transformers
   python src/models/train_hf.py --config configs/default.yaml
   ```

4. **Evaluation**:
   ```bash
   # Evaluate model on test set
   python src/evaluation/evaluate.py --config configs/default.yaml
   
   # Cross-validation
   python src/evaluation/cross_validation.py --config configs/default.yaml
   ```

5. **Create Submission**:
   ```bash
   # Generate submission file
   python submission/make_submission.py --config configs/default.yaml
   
   # Generate and upload submission
   python submission/make_submission.py --config configs/default.yaml --upload --competition [COMPETITION_NAME] --message "Submission description"
   ```

## Configuration

The project uses Hydra/OmegaConf for configuration management. The main configuration file is `configs/default.yaml`, which includes sections for:

- Data paths and parameters
- Preprocessing options
- Model architecture and hyperparameters
- Training settings
- Evaluation metrics
- Logging options

Example:
```yaml
paths:
  data_dir: "data/"
  output_dir: "output/"

data:
  train_file: "train.csv"
  test_file: "test.csv"
  
model:
  type: "resnet"
  hidden_dim: 128
  dropout: 0.2
  
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping: 5
```

## Extending the Boilerplate

### Adding New Models

Add new model architectures in `src/models/model.py`:

```python
class YourNewModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize your model architecture
        
    def forward(self, x):
        # Define forward pass
        return output
```

### Adding Custom Metrics

Add new evaluation metrics in `src/evaluation/evaluate.py`:

```python
def your_custom_metric(y_true, y_pred):
    # Calculate and return your metric
    return score
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the amazing deep learning framework
- Kaggle for hosting competitions and providing API access
- Open source projects that inspired this boilerplate 