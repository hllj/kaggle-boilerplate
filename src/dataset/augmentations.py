"""
Data augmentation module for various data types (images, text, tabular).
"""
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils import resample
import random
import torchaudio
from albumentations.core.transforms_interface import ImageOnlyTransform


class ImageTransforms:
    """
    Image transformations using Albumentations library.
    """

    @staticmethod
    def get_train_transforms(height: int = 224, width: int = 224) -> A.Compose:
        """
        Get training transformations for image data.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Albumentation transforms composition
        """
        return A.Compose([
            A.RandomResizedCrop(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
            ], p=0.25),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            ], p=0.25),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def get_valid_transforms(height: int = 224, width: int = 224) -> A.Compose:
        """
        Get validation transformations for image data.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Albumentation transforms composition
        """
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def get_test_transforms(height: int = 224, width: int = 224) -> A.Compose:
        """
        Get test transformations for image data.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Albumentation transforms composition
        """
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    @staticmethod
    def get_tta_transforms(height: int = 224, width: int = 224) -> List[A.Compose]:
        """
        Get test-time augmentation transformations.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            List of albumentation transforms compositions
        """
        return [
            # Original
            A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Horizontal flip
            A.Compose([
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Vertical flip
            A.Compose([
                A.Resize(height=height, width=width),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Rotation 90
            A.Compose([
                A.Resize(height=height, width=width),
                A.Rotate(limit=(90, 90), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Center crop
            A.Compose([
                A.Resize(height=int(height * 1.2), width=int(width * 1.2)),
                A.CenterCrop(height=height, width=width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]


class TextAugmentations:
    """
    Text augmentation techniques.
    Note: These are simple implementations, for production, consider using
    NLP libraries like nlpaug or textattack.
    """

    @staticmethod
    def random_swap(tokens: List[str], p: float = 0.1) -> List[str]:
        """
        Randomly swap tokens in a text.
        
        Args:
            tokens: List of tokens
            p: Probability of swapping each token
            
        Returns:
            Augmented tokens
        """
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            if np.random.random() < p:
                j = np.random.randint(0, len(new_tokens))
                new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
        return new_tokens

    @staticmethod
    def random_deletion(tokens: List[str], p: float = 0.1) -> List[str]:
        """
        Randomly delete tokens from a text.
        
        Args:
            tokens: List of tokens
            p: Probability of deleting each token
            
        Returns:
            Augmented tokens
        """
        if len(tokens) == 1:
            return tokens
        
        new_tokens = []
        for token in tokens:
            if np.random.random() > p:
                new_tokens.append(token)
                
        if len(new_tokens) == 0:
            return [tokens[np.random.randint(0, len(tokens))]]
            
        return new_tokens

    @staticmethod
    def random_insertion(tokens: List[str], p: float = 0.1) -> List[str]:
        """
        Randomly insert tokens from the original text.
        
        Args:
            tokens: List of tokens
            p: Probability of inserting after each token
            
        Returns:
            Augmented tokens
        """
        new_tokens = tokens.copy()
        for _ in range(int(p * len(tokens))):
            if len(tokens) == 0:
                continue
                
            random_token = tokens[np.random.randint(0, len(tokens))]
            random_pos = np.random.randint(0, len(new_tokens) + 1)
            new_tokens.insert(random_pos, random_token)
            
        return new_tokens

    @staticmethod
    def apply_augmentations(text: str, tokenizer: Callable = str.split, 
                          detokenizer: Callable = lambda x: ' '.join(x),
                          p_swap: float = 0.1, 
                          p_delete: float = 0.1, 
                          p_insert: float = 0.1) -> str:
        """
        Apply text augmentations to a text string.
        
        Args:
            text: Input text
            tokenizer: Function to tokenize text
            detokenizer: Function to detokenize tokens
            p_swap: Probability for token swapping
            p_delete: Probability for token deletion
            p_insert: Probability for token insertion
            
        Returns:
            Augmented text
        """
        tokens = tokenizer(text)
        
        # Apply augmentations with some probability
        if np.random.random() < p_swap:
            tokens = TextAugmentations.random_swap(tokens, p_swap)
            
        if np.random.random() < p_delete:
            tokens = TextAugmentations.random_deletion(tokens, p_delete)
            
        if np.random.random() < p_insert:
            tokens = TextAugmentations.random_insertion(tokens, p_insert)
            
        return detokenizer(tokens)


class TabularAugmentations:
    """
    Tabular data augmentation techniques.
    """

    @staticmethod
    def gaussian_noise(X: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to the features.
        
        Args:
            X: Feature matrix
            scale: Scale of the noise
            
        Returns:
            Augmented feature matrix
        """
        noise = np.random.normal(0, scale, X.shape)
        return X + noise

    @staticmethod
    def bootstrap_sampling(X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate bootstrap samples.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Tuple of augmented features and targets
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        if y is None:
            return X[indices]
        else:
            return X[indices], y[indices]

    @staticmethod
    def smote_like(X: np.ndarray, y: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple SMOTE-like synthetic minority oversampling.
        For real SMOTE, use the imbalanced-learn library.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of nearest neighbors
            
        Returns:
            Tuple of augmented features and targets
        """
        augmented_X = []
        augmented_y = []
        
        # Find minority class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        minority_class = unique_classes[np.argmin(class_counts)]
        majority_class = unique_classes[np.argmax(class_counts)]
        
        # Get indices of minority and majority classes
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Number of samples to generate
        n_to_generate = len(majority_indices) - len(minority_indices)
        
        # Generate synthetic samples
        for _ in range(n_to_generate):
            # Randomly select a minority class sample
            index = np.random.choice(minority_indices)
            sample = X[index]
            
            # Find k nearest neighbors (simplified - just randomly select)
            neighbor_indices = np.random.choice(minority_indices, size=k, replace=True)
            neighbor_index = np.random.choice(neighbor_indices)
            neighbor = X[neighbor_index]
            
            # Generate synthetic sample
            alpha = np.random.random()
            synthetic_sample = sample + alpha * (neighbor - sample)
            
            augmented_X.append(synthetic_sample)
            augmented_y.append(minority_class)
        
        # Combine original and synthetic samples
        augmented_X = np.vstack([X, np.array(augmented_X)])
        augmented_y = np.hstack([y, np.array(augmented_y)])
        
        return augmented_X, augmented_y

    @staticmethod
    def mixup(X: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation.
        
        Args:
            X: Feature matrix
            y: Target vector (one-hot encoded for classification)
            alpha: Mixup parameter
            
        Returns:
            Tuple of augmented features and targets
        """
        batch_size = X.shape[0]
        
        # Sample from Beta distribution
        weight = np.random.beta(alpha, alpha, batch_size)
        
        # Reshape to column vector for element-wise multiplication
        weight = weight.reshape(batch_size, 1)
        
        # Create permutation of indices
        indices = np.random.permutation(batch_size)
        
        # Mixup features
        mixed_X = weight * X + (1 - weight) * X[indices]
        
        # For regression, mixup target values directly
        if len(y.shape) == 1 or y.shape[1] == 1:  # Regression
            weight_y = weight.flatten()
            mixed_y = weight_y * y + (1 - weight_y) * y[indices]
        else:  # Classification with one-hot encoding
            mixed_y = weight * y + (1 - weight) * y[indices]
        
        return mixed_X, mixed_y


# Example of a custom dataset with augmentations
class AugmentedDataset(Dataset):
    """
    Dataset wrapper for applying augmentations.
    
    Attributes:
        dataset: Original dataset
        transform: Transform function to apply
    """
    
    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        """
        Initialize AugmentedDataset.
        
        Args:
            dataset: Original dataset
            transform: Transform function to apply
        """
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with augmentation applied.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with augmented data
        """
        item = self.dataset[idx]
        
        if self.transform:
            if isinstance(item, dict):
                if 'features' in item:
                    item['features'] = self.transform(item['features'])
            else:
                item = self.transform(item)
        
        return item 


def get_transforms(cfg, image_size=None, is_train=True):
    """
    Get transforms for images based on config.
    
    Args:
        cfg: Configuration
        image_size: Image size (optional, overrides config)
        is_train: Whether to get transforms for training or validation
        
    Returns:
        Albumentation transforms
    """
    if image_size is None:
        image_size = cfg.augmentation.image_size
    
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(image_size, image_size),
            A.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.7),
            A.Normalize()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize()
        ])


def normalize_melspec(X, eps=1e-6):
    """
    Normalize mel spectrogram.
    
    Args:
        X: Mel spectrogram tensor
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized mel spectrogram
    """
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V


def mixup(data, targets, alpha=0.5):
    """
    Perform mixup augmentation.
    
    Args:
        data: Batch data
        targets: Batch targets
        alpha: Mixup parameter
        
    Returns:
        Mixed data and targets
    """
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def read_wav(path, sample_rate=32000):
    """
    Read and normalize a wav file.
    
    Args:
        path: Path to wav file
        sample_rate: Target sample rate
        
    Returns:
        Normalized audio tensor
    """
    wav, org_sr = torchaudio.load(path, normalize=True)
    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=sample_rate)
    return wav


def crop_or_pad_wav(wav, duration):
    """
    Crop or pad wav to desired duration.
    
    Args:
        wav: Audio tensor
        duration: Target duration in samples
        
    Returns:
        Audio tensor of length duration
    """
    while wav.size(-1) < duration:
        wav = torch.cat([wav, wav], dim=1)
    wav = wav[:, :duration]
    return wav


class TimeShift(ImageOnlyTransform):
    """
    Time shifting augmentation for spectrograms.
    
    Attributes:
        p: Probability of applying the transform
        max_shift_x: Maximum shift in x direction (time)
    """
    
    def __init__(self, max_shift_x=20, always_apply=False, p=0.5):
        super(TimeShift, self).__init__(always_apply, p)
        self.max_shift_x = max_shift_x
        
    def apply(self, img, **params):
        shift_x = random.randint(-self.max_shift_x, self.max_shift_x)
        if shift_x > 0:
            img = np.roll(img, shift_x, axis=1)
            img[:, :shift_x, :] = 0
        elif shift_x < 0:
            img = np.roll(img, shift_x, axis=1)
            img[:, shift_x:, :] = 0
        return img


class FrequencyMask(ImageOnlyTransform):
    """
    Frequency masking augmentation for spectrograms.
    
    Attributes:
        p: Probability of applying the transform
        max_width: Maximum mask width
        num_masks: Number of masks to apply
    """
    
    def __init__(self, max_width=10, num_masks=1, always_apply=False, p=0.5):
        super(FrequencyMask, self).__init__(always_apply, p)
        self.max_width = max_width
        self.num_masks = num_masks
        
    def apply(self, img, **params):
        height = img.shape[0]
        for _ in range(self.num_masks):
            width = random.randint(1, self.max_width)
            start = random.randint(0, height - width)
            img[start:start+width, :, :] = 0
        return img


class TimeMask(ImageOnlyTransform):
    """
    Time masking augmentation for spectrograms.
    
    Attributes:
        p: Probability of applying the transform
        max_width: Maximum mask width
        num_masks: Number of masks to apply
    """
    
    def __init__(self, max_width=10, num_masks=1, always_apply=False, p=0.5):
        super(TimeMask, self).__init__(always_apply, p)
        self.max_width = max_width
        self.num_masks = num_masks
        
    def apply(self, img, **params):
        width = img.shape[1]
        for _ in range(self.num_masks):
            mask_width = random.randint(1, self.max_width)
            start = random.randint(0, width - mask_width)
            img[:, start:start+mask_width, :] = 0
        return img 