"""
Data loading and preprocessing module.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from src.config import Config
from src.dataset.augmentations import normalize_melspec, read_wav, crop_or_pad_wav


class BirdDataset(Dataset):
    """
    Dataset class for BirdCLEF competition.
    
    Attributes:
        df: DataFrame with metadata
        transform: Albumentation transforms
        add_secondary_labels: Whether to add secondary labels
        mel_transform: Mel spectrogram transform
        db_transform: Amplitude to dB transform
    """
    def __init__(self, 
                 df: pd.DataFrame, 
                 transform=None, 
                 add_secondary_labels=True,
                 mel_spec_params=None,
                 top_db=80,
                 train_duration=None,
                 bird2id=None):
        """
        Initialize BirdDataset.
        
        Args:
            df: DataFrame with metadata
            transform: Albumentation transforms
            add_secondary_labels: Whether to add secondary labels
            mel_spec_params: Parameters for mel spectrogram transform
            top_db: Maximum dB value for amplitude to dB transform
            train_duration: Duration of training samples
            bird2id: Dictionary mapping bird names to indices
        """
        self.df = df
        self.bird2id = bird2id
        self.num_classes = len(bird2id) if bird2id else 0
        self.secondary_coef = 1.0
        self.add_secondary_labels = add_secondary_labels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)
        self.transform = transform
        self.train_duration = train_duration

    def __len__(self):
        return len(self.df)

    def prepare_target(self, primary_label, secondary_labels):
        """
        Prepare target vector with primary and secondary labels.
        
        Args:
            primary_label: Primary label
            secondary_labels: Secondary labels
            
        Returns:
            Target vector
        """
        secondary_labels = eval(secondary_labels)
        target = np.zeros(self.num_classes, dtype=np.float32)
        if primary_label != 'nocall':
            primary_label = self.bird2id[primary_label]
            target[primary_label] = 1.0
            if self.add_secondary_labels:
                for s in secondary_labels:
                    if s != "" and s in self.bird2id.keys():
                        target[self.bird2id[s]] = self.secondary_coef
        target = torch.from_numpy(target).float()
        return target

    def prepare_spec(self, path):
        """
        Prepare mel spectrogram from audio file.
        
        Args:
            path: Path to audio file
            
        Returns:
            Mel spectrogram
        """
        wav = read_wav(path)
        wav = crop_or_pad_wav(wav, self.train_duration)
        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        mel_spectrogram = mel_spectrogram * 255
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1).permute(1, 2, 0).numpy()
        return mel_spectrogram

    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with spectrogram, target, and rating
        """
        path = self.df["path"].iloc[idx]
        primary_label = self.df["primary_label"].iloc[idx]
        secondary_labels = self.df["secondary_labels"].iloc[idx]
        rating = self.df["rating"].iloc[idx]

        spec = self.prepare_spec(path)
        target = self.prepare_target(primary_label, secondary_labels)

        if self.transform is not None:
            res = self.transform(image=spec)
            spec = res['image'].astype(np.float32)
        else:
            spec = spec.astype(np.float32)

        spec = spec.transpose(2, 0, 1)

        return {"spec": spec, "target": target, 'rating': rating}


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


def create_folds(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Create folds for cross-validation.
    
    Args:
        df: DataFrame with data
        cfg: Configuration
        
    Returns:
        DataFrame with fold column
    """
    skf = StratifiedKFold(
        n_splits=cfg.data.num_folds, 
        random_state=cfg.seed, 
        shuffle=True
    )
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["primary_label"].values)):
        df.loc[val_idx, 'fold'] = fold
    return df


def get_bird2id(df: pd.DataFrame, sub_df: pd.DataFrame) -> Dict[str, int]:
    """
    Create mapping from bird names to indices.
    
    Args:
        df: Training DataFrame
        sub_df: Submission DataFrame
        
    Returns:
        Dictionary mapping bird names to indices
    """
    target_columns = sub_df.columns.tolist()[1:]
    bird2id = {b: i for i, b in enumerate(target_columns)}
    return bird2id


def create_data_loaders(cfg: Config, fold: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        cfg: Configuration
        fold: Fold number
        
    Returns:
        Tuple of training and validation data loaders
    """
    # Load data
    df = pd.read_csv(cfg.data.train_file)
    df["path"] = os.path.join(cfg.paths.data_dir, "train_audio/") + df["filename"]
    df["rating"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
    
    # Create folds
    df = create_folds(df, cfg)
    
    # Get bird2id mapping
    sub_df = pd.read_csv(cfg.data.sample_submission_file)
    bird2id = get_bird2id(df, sub_df)
    
    # Split data
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    
    # Create datasets
    train_dataset = BirdDataset(
        df=train_df,
        transform=cfg.augmentation.train_transform,
        add_secondary_labels=True,
        mel_spec_params=cfg.data.mel_spec_params,
        top_db=cfg.data.top_db,
        train_duration=cfg.data.train_duration,
        bird2id=bird2id
    )
    
    val_dataset = BirdDataset(
        df=val_df,
        transform=cfg.augmentation.val_transform,
        add_secondary_labels=True,
        mel_spec_params=cfg.data.mel_spec_params,
        top_db=cfg.data.top_db,
        train_duration=cfg.data.train_duration,
        bird2id=bird2id
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.resources.num_workers,
        pin_memory=cfg.resources.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.resources.num_workers,
        pin_memory=cfg.resources.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


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
    train_dataset = BirdDataset(X_train, y_train)
    test_dataset = BirdDataset(X_test)
    
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