"""
Data loading and preprocessing module.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from src.config import Config


class KaggleDataset(Dataset):
    """
    PyTorch Dataset for Kaggle competition.
    
    Attributes:
        features: Feature matrix
        targets: Target vector
        transform: Optional transformation function
    """
    
    def __init__(self, 
                features: np.ndarray, 
                targets: Optional[np.ndarray] = None, 
                transform: Optional[callable] = None):
        """
        Initialize KaggleDataset.
        
        Args:
            features: Feature matrix
            targets: Target vector (optional for test set)
            transform: Optional transformation function
        """
        self.features = features
        self.targets = targets
        self.transform = transform
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with features and optionally targets
        """
        x = self.features[idx]
        
        # Apply transform if provided
        if self.transform:
            x = self.transform(x)
        
        # Convert to tensor if not already
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Create output dictionary
        output = {'features': x}
        
        # Add target if available
        if self.targets is not None:
            y = self.targets[idx]
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
            output['target'] = y
        
        return output


def preprocess_data(train_df: pd.DataFrame, 
                   test_df: pd.DataFrame, 
                   cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for training and testing.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        cfg: Configuration
        
    Returns:
        Tuple of preprocessed X_train, y_train, and X_test
    """
    # Extract target
    y_train = train_df[cfg.data.target_column].values
    
    # Drop target from training data
    X_train = train_df.drop(columns=[cfg.data.target_column])
    X_test = test_df.copy()
    
    # Get feature columns
    numerical_cols = cfg.data.features.numerical
    categorical_cols = cfg.data.features.categorical
    
    # If no features specified, use all columns
    if not numerical_cols and not categorical_cols:
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values
    if cfg.preprocessing.handle_missing:
        # Fill numerical missing values with median
        for col in numerical_cols:
            if col in X_train.columns:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if col in X_train.columns:
                mode_val = X_train[col].mode()[0]
                X_train[col] = X_train[col].fillna(mode_val)
                X_test[col] = X_test[col].fillna(mode_val)
    
    # Encode categorical variables
    if cfg.preprocessing.categorical_encoding == 'label':
        for col in categorical_cols:
            if col in X_train.columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
    
    elif cfg.preprocessing.categorical_encoding == 'onehot':
        for col in categorical_cols:
            if col in X_train.columns:
                # Create one-hot encoder
                ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                
                # Fit and transform training data
                train_ohe = ohe.fit_transform(X_train[[col]])
                
                # Transform test data
                test_ohe = ohe.transform(X_test[[col]])
                
                # Create dataframes with one-hot encoded values
                train_ohe_df = pd.DataFrame(
                    train_ohe, 
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                    index=X_train.index
                )
                
                test_ohe_df = pd.DataFrame(
                    test_ohe, 
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                    index=X_test.index
                )
                
                # Concatenate one-hot encoded columns to original dataframes
                X_train = pd.concat([X_train.drop(columns=[col]), train_ohe_df], axis=1)
                X_test = pd.concat([X_test.drop(columns=[col]), test_ohe_df], axis=1)
    
    # Scale numerical features
    if cfg.preprocessing.scaling == 'standard':
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    elif cfg.preprocessing.scaling == 'minmax':
        scaler = MinMaxScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Feature selection
    if cfg.preprocessing.feature_selection.enabled:
        # Implement feature selection based on method
        pass
    
    return X_train.values, y_train, X_test.values


def create_data_loaders(X_train: np.ndarray, 
                       y_train: np.ndarray, 
                       X_test: np.ndarray,
                       cfg: Config,
                       validation_split: bool = True) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        cfg: Configuration
        validation_split: Whether to split training data into train and validation sets
        
    Returns:
        Dictionary of DataLoaders for train, validation, and test sets
    """
    # Create datasets
    train_dataset = KaggleDataset(X_train, y_train)
    test_dataset = KaggleDataset(X_test)
    
    # Create dict to store data loaders
    data_loaders = {}
    
    if validation_split:
        # Split indices for train and validation
        indices = list(range(len(train_dataset)))
        
        # Use stratified split if target is binary or multiclass
        if np.issubdtype(y_train.dtype, np.integer) or len(np.unique(y_train)) < 10:
            # Stratified split
            train_idx, valid_idx = train_test_split(
                indices, 
                test_size=cfg.data.test_size,
                random_state=cfg.data.random_state,
                stratify=y_train
            )
        else:
            # Random split
            train_idx, valid_idx = train_test_split(
                indices, 
                test_size=cfg.data.test_size,
                random_state=cfg.data.random_state
            )
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # Create data loaders
        data_loaders['train'] = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=train_sampler,
            num_workers=cfg.resources.num_workers,
            pin_memory=cfg.resources.pin_memory
        )
        
        data_loaders['val'] = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=valid_sampler,
            num_workers=cfg.resources.num_workers,
            pin_memory=cfg.resources.pin_memory
        )
    else:
        # Use all data for training
        data_loaders['train'] = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.resources.num_workers,
            pin_memory=cfg.resources.pin_memory
        )
    
    # Test data loader
    data_loaders['test'] = DataLoader(
        test_dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=cfg.resources.num_workers,
        pin_memory=cfg.resources.pin_memory
    )
    
    return data_loaders


def create_kfold_dataloaders(X_train: np.ndarray, 
                            y_train: np.ndarray, 
                            X_test: np.ndarray,
                            cfg: Config) -> List[Dict[str, DataLoader]]:
    """
    Create K-Fold cross-validation DataLoaders.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        cfg: Configuration
        
    Returns:
        List of dictionaries containing DataLoaders for each fold
    """
    # Create datasets
    train_dataset = KaggleDataset(X_train, y_train)
    test_dataset = KaggleDataset(X_test)
    
    # Create K-Fold cross-validation
    if np.issubdtype(y_train.dtype, np.integer) or len(np.unique(y_train)) < 10:
        # Use stratified k-fold for classification
        kfold = StratifiedKFold(n_splits=cfg.data.num_folds, shuffle=True, random_state=cfg.data.random_state)
        splits = kfold.split(X_train, y_train)
    else:
        # Use regular k-fold for regression
        kfold = KFold(n_splits=cfg.data.num_folds, shuffle=True, random_state=cfg.data.random_state)
        splits = kfold.split(X_train)
    
    # List to store dataloaders for each fold
    fold_data_loaders = []
    
    for train_idx, valid_idx in splits:
        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # Create data loaders for this fold
        fold_loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                sampler=train_sampler,
                num_workers=cfg.resources.num_workers,
                pin_memory=cfg.resources.pin_memory
            ),
            'val': DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                sampler=valid_sampler,
                num_workers=cfg.resources.num_workers,
                pin_memory=cfg.resources.pin_memory
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=cfg.inference.batch_size,
                shuffle=False,
                num_workers=cfg.resources.num_workers,
                pin_memory=cfg.resources.pin_memory
            )
        }
        
        # Add to list of fold dataloaders
        fold_data_loaders.append(fold_loaders)
    
    return fold_data_loaders 