"""
PyTorch model definitions.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from transformers import AutoModel

from src.config import Config


class TabularMLP(nn.Module):
    """
    Multi-layer perceptron for tabular data.
    
    Attributes:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        output_dim: Output dimension
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        activation: Activation function
    """
    
    def __init__(self, 
                input_dim: int, 
                hidden_dims: List[int] = [256, 128, 64], 
                output_dim: int = 1, 
                dropout: float = 0.5,
                batch_norm: bool = True,
                activation: str = 'relu'):
        """
        Initialize TabularMLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Create activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activation {activation} not supported")
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for image data.
    
    Attributes:
        backbone: ResNet backbone
        fc: Fully connected head
        dropout: Dropout layer
    """
    
    def __init__(self, 
                num_classes: int = 1, 
                backbone: str = 'resnet50', 
                pretrained: bool = True,
                dropout: float = 0.5):
        """
        Initialize ResNetClassifier.
        
        Args:
            num_classes: Number of output classes
            backbone: ResNet backbone ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # Get feature dimension
        in_features = self.backbone.fc.in_features
        
        # Replace fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Create new head
        self.dropout = nn.Dropout(dropout)
        
        # For binary classification with sigmoid, use 1 output unit
        if num_classes == 1:
            self.fc = nn.Linear(in_features, 1)
        else:
            self.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.fc(features)
        
        return logits


class TransformerTextClassifier(nn.Module):
    """
    Transformer-based text classifier.
    
    Attributes:
        transformer: Transformer model
        dropout: Dropout layer
        fc: Fully connected head
    """
    
    def __init__(self, 
                model_name: str = 'bert-base-uncased', 
                num_classes: int = 1,
                dropout: float = 0.3,
                freeze_base: bool = False):
        """
        Initialize TransformerTextClassifier.
        
        Args:
            model_name: Name of the pretrained transformer model
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_base: Whether to freeze the transformer base
        """
        super().__init__()
        
        # Load transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Get hidden size of the model
        hidden_size = self.transformer.config.hidden_size
        
        # Create classification head
        self.dropout = nn.Dropout(dropout)
        
        # For binary classification with sigmoid, use 1 output unit
        if num_classes == 1:
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, 
               input_ids: torch.Tensor, 
               attention_mask: torch.Tensor,
               token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            
        Returns:
            Output tensor
        """
        # Get transformer outputs
        if token_type_ids is not None:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get pooled output (CLS token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
        return logits


class TabNetModel(nn.Module):
    """
    Simplified TabNet for tabular data.
    
    This is a simplified implementation for demonstration.
    For production, consider using the official implementation or PyTorch Tabular.
    
    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension
        n_d: Feature dimension in decision step
        n_a: Feature dimension in attention step
        n_steps: Number of decision steps
        gamma: Information routing parameter
        cat_idxs: List of categorical feature indices
        cat_dims: List of categorical feature dimensions
        n_independent: Number of independent feature transformers
        n_shared: Number of shared feature transformers
    """
    
    def __init__(self, 
                input_dim: int,
                output_dim: int = 1,
                n_d: int = 64,
                n_a: int = 64,
                n_steps: int = 3,
                gamma: float = 1.3,
                cat_idxs: List[int] = None,
                cat_dims: List[int] = None,
                n_independent: int = 2,
                n_shared: int = 2):
        """
        Initialize TabNetModel.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_d: Feature dimension in decision step
            n_a: Feature dimension in attention step
            n_steps: Number of decision steps
            gamma: Information routing parameter
            cat_idxs: List of categorical feature indices
            cat_dims: List of categorical feature dimensions
            n_independent: Number of independent feature transformers
            n_shared: Number of shared feature transformers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Initialize feature preprocessing (simplified)
        self.initial_fc = nn.Linear(input_dim, n_d + n_a)
        self.initial_bn = nn.BatchNorm1d(n_d + n_a)
        
        # Initialize feature transformers
        self.independent_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_d + n_a, n_d + n_a),
                nn.BatchNorm1d(n_d + n_a),
                nn.ReLU()
            ) for _ in range(n_independent)
        ])
        
        self.shared_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_d + n_a, n_d + n_a),
                nn.BatchNorm1d(n_d + n_a),
                nn.ReLU()
            ) for _ in range(n_shared)
        ])
        
        # Initialize attentive transformer and feature transformer (simplified)
        self.attentive_transformer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.Sigmoid()
            ) for _ in range(n_steps)
        ])
        
        # Output layer for final prediction
        self.final_fc = nn.Linear(n_d * n_steps, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        batch_size = x.size(0)
        
        # Initial feature transformation
        x_processed = self.initial_fc(x)
        x_processed = self.initial_bn(x_processed)
        x_processed = F.relu(x_processed)
        
        # Apply feature transformations
        for independent, shared in zip(self.independent_fcs, self.shared_fcs):
            x_processed = independent(x_processed) + shared(x_processed)
        
        # TabNet decision step
        steps_output = []
        prior = torch.ones(batch_size, self.input_dim).to(x.device)
        
        for step in range(self.n_steps):
            # Split features for decision and attention
            d = x_processed[:, :self.n_d]
            a = x_processed[:, self.n_d:]
            
            # Calculate attention mask
            mask = self.attentive_transformer[step](a)
            mask = mask * prior
            prior = prior * (self.gamma - mask)
            
            # Apply mask to input
            masked_x = x * mask
            
            # Process masked input (simplified)
            masked_x_processed = self.initial_fc(masked_x)
            masked_x_processed = self.initial_bn(masked_x_processed)
            masked_x_processed = F.relu(masked_x_processed)
            
            for independent, shared in zip(self.independent_fcs, self.shared_fcs):
                masked_x_processed = independent(masked_x_processed) + shared(masked_x_processed)
            
            # Get decision output
            d_masked = masked_x_processed[:, :self.n_d]
            steps_output.append(d_masked)
        
        # Concatenate step outputs
        out = torch.cat(steps_output, dim=1)
        
        # Final output layer
        out = self.final_fc(out)
        
        return out


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    
    Attributes:
        p: Pooling parameter (learnable)
        eps: Small value to prevent numerical issues
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(
            1.0 / self.p)
        x = x.view(bs, ch)
        return x


class BirdCLEFModel(nn.Module):
    """
    CNN model with backbone from timm and GeM pooling for BirdCLEF competition.
    
    Attributes:
        backbone: CNN backbone from timm
        global_pools: Pooling layers
        neck: Batch normalization layer
        head: Fully connected head
    """
    def __init__(self, 
                 num_classes: int, 
                 backbone: str = 'eca_nfnet_l0', 
                 pretrained: bool = True):
        """
        Initialize BirdCLEFModel.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone model name from timm
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        out_indices = (3, 4)
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
            out_indices=out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()

        self.global_pools = nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = sum(feature_dims)
        self.neck = nn.BatchNorm1d(self.mid_features)
        self.head = nn.Linear(self.mid_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (spectrograms)
            
        Returns:
            Output logits
        """
        ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        x = self.neck(h)
        x = self.head(x)
        return x


def get_model(cfg: Config) -> nn.Module:
    """
    Get model based on configuration.
    
    Args:
        cfg: Configuration
        
    Returns:
        PyTorch model
    """
    model_name = cfg.model.name.lower()
    
    if model_name == 'tabular_mlp':
        return TabularMLP(
            input_dim=cfg.model.input_dim,
            hidden_dims=[cfg.model.hidden_dim] * 3,
            output_dim=cfg.model.num_classes,
            dropout=cfg.model.dropout
        )
    elif model_name == 'resnet':
        return ResNetClassifier(
            num_classes=cfg.model.num_classes,
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            dropout=cfg.model.dropout
        )
    elif model_name == 'transformer':
        return TransformerTextClassifier(
            model_name=cfg.model.backbone,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout
        )
    elif model_name == 'tabnet':
        return TabNetModel(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.num_classes
        )
    elif model_name == 'birdclef':
        return BirdCLEFModel(
            num_classes=cfg.model.num_classes,
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained
        )
    else:
        raise ValueError(f"Model {model_name} not supported") 