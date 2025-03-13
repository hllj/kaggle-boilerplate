"""
Feature engineering module.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression

from src.config import Config


def create_date_features(df: pd.DataFrame, 
                        date_column: str, 
                        drop_original: bool = True) -> pd.DataFrame:
    """
    Create date features from a date column.
    
    Args:
        df: DataFrame containing the date column
        date_column: Name of the date column
        drop_original: Whether to drop the original date column
        
    Returns:
        DataFrame with new date features
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract date features
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df[f'{date_column}_dayofyear'] = df[date_column].dt.dayofyear
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df[f'{date_column}_is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df[f'{date_column}_is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df[f'{date_column}_is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df[f'{date_column}_is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    
    # Drop original date column if requested
    if drop_original:
        df = df.drop(columns=[date_column])
    
    return df


def create_aggregation_features(df: pd.DataFrame, 
                               group_cols: List[str], 
                               agg_cols: List[str],
                               agg_funcs: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create aggregation features by grouping on specified columns.
    
    Args:
        df: DataFrame
        group_cols: Columns to group by
        agg_cols: Columns to aggregate
        agg_funcs: Aggregation functions to apply
        
    Returns:
        DataFrame with new aggregation features
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Create aggregation dictionary
    agg_dict = {col: agg_funcs for col in agg_cols}
    
    # Group and aggregate
    agg_df = df.groupby(group_cols).agg(agg_dict)
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    # Reset index to get group columns back
    agg_df = agg_df.reset_index()
    
    # Merge with original dataframe
    df = df.merge(agg_df, on=group_cols, how='left')
    
    return df


def create_interaction_features(df: pd.DataFrame, 
                               feature_pairs: List[Tuple[str, str]],
                               operations: List[str] = ['sum', 'diff', 'product', 'ratio']) -> pd.DataFrame:
    """
    Create interaction features between pairs of features.
    
    Args:
        df: DataFrame
        feature_pairs: List of tuples containing pairs of features to interact
        operations: List of operations to apply (sum, diff, product, ratio)
        
    Returns:
        DataFrame with new interaction features
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Create interaction features
    for col1, col2 in feature_pairs:
        if 'sum' in operations:
            df[f'{col1}_{col2}_sum'] = df[col1] + df[col2]
        
        if 'diff' in operations:
            df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
        
        if 'product' in operations:
            df[f'{col1}_{col2}_product'] = df[col1] * df[col2]
        
        if 'ratio' in operations:
            # Avoid division by zero
            df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
    
    return df


def create_binning_features(df: pd.DataFrame, 
                           cols: List[str], 
                           n_bins: int = 10,
                           strategy: str = 'uniform') -> pd.DataFrame:
    """
    Create binning features from numerical columns.
    
    Args:
        df: DataFrame
        cols: Columns to bin
        n_bins: Number of bins
        strategy: Binning strategy ('uniform', 'quantile')
        
    Returns:
        DataFrame with new binning features
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Create binning features
    for col in cols:
        if strategy == 'uniform':
            df[f'{col}_bin'] = pd.cut(df[col], bins=n_bins, labels=False)
        else:  # quantile binning
            df[f'{col}_bin'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
    
    return df


def create_pca_features(X_train: np.ndarray, 
                       X_test: np.ndarray, 
                       n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create PCA features.
    
    Args:
        X_train: Training features
        X_test: Test features
        n_components: Number of PCA components
        
    Returns:
        Tuple of transformed training and test features
    """
    # Create PCA object
    pca = PCA(n_components=n_components)
    
    # Fit and transform training data
    X_train_pca = pca.fit_transform(X_train)
    
    # Transform test data
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_test_pca


def select_features(X_train: np.ndarray, 
                   y_train: np.ndarray, 
                   X_test: np.ndarray, 
                   method: str = 'mutual_info',
                   k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select k best features.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        method: Feature selection method ('mutual_info', 'chi2', 'f_value')
        k: Number of features to select
        
    Returns:
        Tuple of transformed training and test features
    """
    # Create selector based on method
    if method == 'mutual_info':
        if len(np.unique(y_train)) < 10:  # Classification
            selector = SelectKBest(mutual_info_regression, k=k)
        else:  # Regression
            selector = SelectKBest(mutual_info_regression, k=k)
    elif method == 'chi2':
        # Ensure non-negative values for chi2
        X_train = np.abs(X_train)
        X_test = np.abs(X_test)
        selector = SelectKBest(chi2, k=k)
    elif method == 'f_value':
        selector = SelectKBest(f_regression, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit and transform training data
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected


def feature_engineering_pipeline(train_df: pd.DataFrame, 
                                test_df: pd.DataFrame, 
                                cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply feature engineering pipeline to training and test data.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        cfg: Configuration
        
    Returns:
        Tuple of transformed training and test dataframes
    """
    # Make copies of dataframes
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Apply feature engineering steps here based on config
    # For example:
    
    # 1. Create date features if date columns are specified
    # date_columns = cfg.feature_engineering.date_columns
    # if date_columns:
    #     for col in date_columns:
    #         train_df = create_date_features(train_df, col)
    #         test_df = create_date_features(test_df, col)
    
    # 2. Create aggregation features if specified
    # if cfg.feature_engineering.aggregations.enabled:
    #     group_cols = cfg.feature_engineering.aggregations.group_cols
    #     agg_cols = cfg.feature_engineering.aggregations.agg_cols
    #     agg_funcs = cfg.feature_engineering.aggregations.agg_funcs
    #     train_df = create_aggregation_features(train_df, group_cols, agg_cols, agg_funcs)
    #     test_df = create_aggregation_features(test_df, group_cols, agg_cols, agg_funcs)
    
    # 3. Create interaction features if specified
    # if cfg.feature_engineering.interactions.enabled:
    #     feature_pairs = cfg.feature_engineering.interactions.feature_pairs
    #     operations = cfg.feature_engineering.interactions.operations
    #     train_df = create_interaction_features(train_df, feature_pairs, operations)
    #     test_df = create_interaction_features(test_df, feature_pairs, operations)
    
    # Add more feature engineering steps as needed
    
    return train_df, test_df 