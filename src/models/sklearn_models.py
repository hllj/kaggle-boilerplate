"""
Scikit-learn models module for Kaggle competitions.
"""
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    GradientBoostingClassifier, 
    GradientBoostingRegressor,
    AdaBoostClassifier, 
    AdaBoostRegressor,
    VotingClassifier, 
    VotingRegressor,
    StackingClassifier, 
    StackingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, 
    Ridge, 
    Lasso, 
    ElasticNet,
    SGDClassifier, 
    SGDRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from src.config import Config


class SklearnModel:
    """
    Base class for scikit-learn models, providing common functionality
    for saving, loading, training, and inference.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SklearnModel.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.target_name = None
        self.is_classifier = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None, 
            categorical_features: Optional[List[int]] = None) -> 'SklearnModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            feature_names: Optional list of feature names
            categorical_features: Optional list of indices of categorical features
            
        Returns:
            Fitted model
        """
        self.feature_names = feature_names
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        return np.zeros(len(X))  # Placeholder to be overridden

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted or not self.is_classifier:
            raise ValueError("Model is not a fitted classifier.")
        
        return np.zeros((len(X), 2))  # Placeholder to be overridden

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'SklearnModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importances if available.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        
        # If feature names are not available, return just the importances
        if self.feature_names is None:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        # Otherwise, map feature names to importances
        return {name: imp for name, imp in zip(self.feature_names, importances)}


class LinearModel(SklearnModel):
    """Linear models from scikit-learn."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a linear model.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        super().__init__(config)
        
        model_type = config.get('type', 'logistic')
        self.is_classifier = model_type in ['logistic', 'sgd_classifier', 'lda']
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                C=config.get('C', 1.0),
                penalty=config.get('penalty', 'l2'),
                solver=config.get('solver', 'lbfgs'),
                max_iter=config.get('max_iter', 1000),
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'ridge':
            self.model = Ridge(
                alpha=config.get('alpha', 1.0),
                solver=config.get('solver', 'auto'),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'lasso':
            self.model = Lasso(
                alpha=config.get('alpha', 1.0),
                random_state=config.get('random_state', 42),
                max_iter=config.get('max_iter', 1000)
            )
        elif model_type == 'elasticnet':
            self.model = ElasticNet(
                alpha=config.get('alpha', 1.0),
                l1_ratio=config.get('l1_ratio', 0.5),
                random_state=config.get('random_state', 42),
                max_iter=config.get('max_iter', 1000)
            )
        elif model_type == 'sgd_classifier':
            self.model = SGDClassifier(
                loss=config.get('loss', 'hinge'),
                penalty=config.get('penalty', 'l2'),
                alpha=config.get('alpha', 0.0001),
                max_iter=config.get('max_iter', 1000),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'sgd_regressor':
            self.model = SGDRegressor(
                loss=config.get('loss', 'squared_error'),
                penalty=config.get('penalty', 'l2'),
                alpha=config.get('alpha', 0.0001),
                max_iter=config.get('max_iter', 1000),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'lda':
            self.model = LinearDiscriminantAnalysis(
                solver=config.get('solver', 'svd'),
                shrinkage=config.get('shrinkage', None)
            )
        else:
            raise ValueError(f"Unknown linear model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None, 
            categorical_features: Optional[List[int]] = None) -> 'LinearModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            feature_names: Optional list of feature names
            categorical_features: Optional list of indices of categorical features
            
        Returns:
            Fitted model
        """
        self.model.fit(X, y)
        return super().fit(X, y, feature_names, categorical_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_classifier:
            raise ValueError("This model does not support probability predictions.")
        
        return self.model.predict_proba(X)


class TreeModel(SklearnModel):
    """Tree-based models from scikit-learn."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a tree-based model.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        super().__init__(config)
        
        model_type = config.get('type', 'random_forest_classifier')
        self.is_classifier = 'classifier' in model_type
        
        if model_type == 'decision_tree_classifier':
            self.model = DecisionTreeClassifier(
                max_depth=config.get('max_depth', None),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                criterion=config.get('criterion', 'gini'),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'decision_tree_regressor':
            self.model = DecisionTreeRegressor(
                max_depth=config.get('max_depth', None),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                criterion=config.get('criterion', 'squared_error'),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'random_forest_classifier':
            self.model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', None),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                criterion=config.get('criterion', 'gini'),
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'random_forest_regressor':
            self.model = RandomForestRegressor(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', None),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                criterion=config.get('criterion', 'squared_error'),
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'gradient_boosting_classifier':
            self.model = GradientBoostingClassifier(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 3),
                subsample=config.get('subsample', 1.0),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'gradient_boosting_regressor':
            self.model = GradientBoostingRegressor(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 3),
                subsample=config.get('subsample', 1.0),
                random_state=config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown tree model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None, 
            categorical_features: Optional[List[int]] = None) -> 'TreeModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            feature_names: Optional list of feature names
            categorical_features: Optional list of indices of categorical features
            
        Returns:
            Fitted model
        """
        self.model.fit(X, y)
        return super().fit(X, y, feature_names, categorical_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_classifier:
            raise ValueError("This model does not support probability predictions.")
        
        return self.model.predict_proba(X)


class BoostingModel(SklearnModel):
    """Boosting models including XGBoost, LightGBM, and CatBoost."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a boosting model.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        super().__init__(config)
        
        model_type = config.get('type', 'xgboost_classifier')
        self.is_classifier = 'classifier' in model_type
        
        if model_type == 'xgboost_classifier':
            self.model = XGBClassifier(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 6),
                subsample=config.get('subsample', 1.0),
                colsample_bytree=config.get('colsample_bytree', 1.0),
                reg_alpha=config.get('reg_alpha', 0),
                reg_lambda=config.get('reg_lambda', 1),
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'xgboost_regressor':
            self.model = XGBRegressor(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 6),
                subsample=config.get('subsample', 1.0),
                colsample_bytree=config.get('colsample_bytree', 1.0),
                reg_alpha=config.get('reg_alpha', 0),
                reg_lambda=config.get('reg_lambda', 1),
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'lightgbm_classifier':
            self.model = LGBMClassifier(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', -1),
                num_leaves=config.get('num_leaves', 31),
                subsample=config.get('subsample', 1.0),
                colsample_bytree=config.get('colsample_bytree', 1.0),
                reg_alpha=config.get('reg_alpha', 0),
                reg_lambda=config.get('reg_lambda', 0),
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'lightgbm_regressor':
            self.model = LGBMRegressor(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', -1),
                num_leaves=config.get('num_leaves', 31),
                subsample=config.get('subsample', 1.0),
                colsample_bytree=config.get('colsample_bytree', 1.0),
                reg_alpha=config.get('reg_alpha', 0),
                reg_lambda=config.get('reg_lambda', 0),
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'catboost_classifier':
            self.model = CatBoostClassifier(
                iterations=config.get('iterations', 100),
                learning_rate=config.get('learning_rate', 0.1),
                depth=config.get('depth', 6),
                l2_leaf_reg=config.get('l2_leaf_reg', 3),
                random_seed=config.get('random_state', 42),
                thread_count=config.get('n_jobs', -1),
                verbose=config.get('verbose', False)
            )
        elif model_type == 'catboost_regressor':
            self.model = CatBoostRegressor(
                iterations=config.get('iterations', 100),
                learning_rate=config.get('learning_rate', 0.1),
                depth=config.get('depth', 6),
                l2_leaf_reg=config.get('l2_leaf_reg', 3),
                random_seed=config.get('random_state', 42),
                thread_count=config.get('n_jobs', -1),
                verbose=config.get('verbose', False)
            )
        else:
            raise ValueError(f"Unknown boosting model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None, 
            categorical_features: Optional[List[int]] = None) -> 'BoostingModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            feature_names: Optional list of feature names
            categorical_features: Optional list of indices of categorical features
            
        Returns:
            Fitted model
        """
        fit_params = {}
        
        if feature_names is not None:
            if isinstance(self.model, (XGBClassifier, XGBRegressor)):
                fit_params['feature_names'] = feature_names
            elif isinstance(self.model, (LGBMClassifier, LGBMRegressor)):
                fit_params['feature_name'] = feature_names
            
        if categorical_features is not None:
            if isinstance(self.model, (CatBoostClassifier, CatBoostRegressor)):
                fit_params['cat_features'] = categorical_features
            elif isinstance(self.model, (LGBMClassifier, LGBMRegressor)):
                fit_params['categorical_feature'] = categorical_features
        
        # Handle evaluation sets for early stopping
        if self.config.get('early_stopping_rounds') is not None:
            if self.config.get('eval_set') is not None:
                fit_params['eval_set'] = self.config['eval_set']
                fit_params['early_stopping_rounds'] = self.config['early_stopping_rounds']
        
        self.model.fit(X, y, **fit_params)
        return super().fit(X, y, feature_names, categorical_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_classifier:
            raise ValueError("This model does not support probability predictions.")
        
        return self.model.predict_proba(X)


class OtherModel(SklearnModel):
    """Other models from scikit-learn."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a model.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        super().__init__(config)
        
        model_type = config.get('type', 'svc')
        self.is_classifier = model_type in ['svc', 'knn_classifier', 'gaussian_nb', 
                                           'mlp_classifier', 'gaussian_process_classifier']
        
        if model_type == 'svc':
            self.model = SVC(
                C=config.get('C', 1.0),
                kernel=config.get('kernel', 'rbf'),
                gamma=config.get('gamma', 'scale'),
                probability=config.get('probability', True),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'svr':
            self.model = SVR(
                C=config.get('C', 1.0),
                kernel=config.get('kernel', 'rbf'),
                gamma=config.get('gamma', 'scale')
            )
        elif model_type == 'knn_classifier':
            self.model = KNeighborsClassifier(
                n_neighbors=config.get('n_neighbors', 5),
                weights=config.get('weights', 'uniform'),
                p=config.get('p', 2),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'knn_regressor':
            self.model = KNeighborsRegressor(
                n_neighbors=config.get('n_neighbors', 5),
                weights=config.get('weights', 'uniform'),
                p=config.get('p', 2),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'gaussian_nb':
            self.model = GaussianNB(
                var_smoothing=config.get('var_smoothing', 1e-9)
            )
        elif model_type == 'mlp_classifier':
            self.model = MLPClassifier(
                hidden_layer_sizes=config.get('hidden_layer_sizes', (100,)),
                activation=config.get('activation', 'relu'),
                solver=config.get('solver', 'adam'),
                alpha=config.get('alpha', 0.0001),
                learning_rate=config.get('learning_rate', 'constant'),
                max_iter=config.get('max_iter', 200),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'mlp_regressor':
            self.model = MLPRegressor(
                hidden_layer_sizes=config.get('hidden_layer_sizes', (100,)),
                activation=config.get('activation', 'relu'),
                solver=config.get('solver', 'adam'),
                alpha=config.get('alpha', 0.0001),
                learning_rate=config.get('learning_rate', 'constant'),
                max_iter=config.get('max_iter', 200),
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'gaussian_process_classifier':
            self.model = GaussianProcessClassifier(
                random_state=config.get('random_state', 42),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'gaussian_process_regressor':
            self.model = GaussianProcessRegressor(
                random_state=config.get('random_state', 42)
            )
        elif model_type == 'kmeans':
            self.model = KMeans(
                n_clusters=config.get('n_clusters', 8),
                init=config.get('init', 'k-means++'),
                n_init=config.get('n_init', 10),
                max_iter=config.get('max_iter', 300),
                random_state=config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None, 
            categorical_features: Optional[List[int]] = None) -> 'OtherModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            feature_names: Optional list of feature names
            categorical_features: Optional list of indices of categorical features
            
        Returns:
            Fitted model
        """
        if model_type == 'kmeans':  # Unsupervised
            self.model.fit(X)
        else:  # Supervised
            self.model.fit(X, y)
        
        return super().fit(X, y, feature_names, categorical_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if hasattr(self.model, 'type') and self.model.type == 'kmeans':
            return self.model.predict(X)  # Cluster assignments
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_classifier:
            raise ValueError("This model does not support probability predictions.")
        
        return self.model.predict_proba(X)


class EnsembleModel(SklearnModel):
    """Ensemble models from scikit-learn."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize an ensemble model.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        super().__init__(config)
        
        model_type = config.get('type', 'voting_classifier')
        self.is_classifier = 'classifier' in model_type
        
        # Extract base estimators from config
        self.base_estimators = config.get('estimators', [])
        estimators = []
        
        # Initialize base estimators
        for i, est_config in enumerate(self.base_estimators):
            est_name = est_config.get('name', f'estimator_{i}')
            est_type = est_config.get('type', '')
            est_params = est_config.get('params', {})
            
            if 'linear' in est_type:
                estimator = LinearModel(est_params).model
            elif 'tree' in est_type:
                estimator = TreeModel(est_params).model
            elif 'boost' in est_type:
                estimator = BoostingModel(est_params).model
            else:
                estimator = OtherModel(est_params).model
            
            estimators.append((est_name, estimator))
        
        if model_type == 'voting_classifier':
            self.model = VotingClassifier(
                estimators=estimators,
                voting=config.get('voting', 'hard'),
                weights=config.get('weights', None),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'voting_regressor':
            self.model = VotingRegressor(
                estimators=estimators,
                weights=config.get('weights', None),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'stacking_classifier':
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=config.get('final_estimator', LogisticRegression()),
                cv=config.get('cv', 5),
                stack_method=config.get('stack_method', 'auto'),
                n_jobs=config.get('n_jobs', -1)
            )
        elif model_type == 'stacking_regressor':
            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=config.get('final_estimator', Ridge()),
                cv=config.get('cv', 5),
                n_jobs=config.get('n_jobs', -1)
            )
        else:
            raise ValueError(f"Unknown ensemble model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None, 
            categorical_features: Optional[List[int]] = None) -> 'EnsembleModel':
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training targets
            feature_names: Optional list of feature names
            categorical_features: Optional list of indices of categorical features
            
        Returns:
            Fitted model
        """
        self.model.fit(X, y)
        return super().fit(X, y, feature_names, categorical_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions on new data (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_classifier:
            raise ValueError("This model does not support probability predictions.")
        
        return self.model.predict_proba(X)


def get_sklearn_model(config: Dict[str, Any]) -> SklearnModel:
    """
    Factory function to get a scikit-learn model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_class = config.get('class', 'linear')
    
    if model_class == 'linear':
        return LinearModel(config)
    elif model_class == 'tree':
        return TreeModel(config)
    elif model_class == 'boosting':
        return BoostingModel(config)
    elif model_class == 'ensemble':
        return EnsembleModel(config)
    elif model_class == 'other':
        return OtherModel(config)
    else:
        raise ValueError(f"Unknown model class: {model_class}") 