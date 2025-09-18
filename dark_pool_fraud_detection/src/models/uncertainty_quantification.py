"""
Uncertainty quantification using Bayesian Neural Networks and Spectral-normalized Neural Gaussian Processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 prior_std: float = 1.0):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            prior_std: Prior standard deviation
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters (mean and log variance)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        # Initialize weight means
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)
        
        # Initialize log variances (small values)
        nn.init.constant_(self.weight_logvar, -3.0)
        nn.init.constant_(self.bias_logvar, -3.0)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass through Bayesian linear layer.
        
        Args:
            x: Input tensor [batch_size, in_features]
            sample: Whether to sample from posterior
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        if sample:
            # Sample weights and biases from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean parameters
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate KL divergence between posterior and prior.
        
        Returns:
            KL divergence value
        """
        # KL divergence for weights
        weight_kl = 0.5 * torch.sum(
            self.weight_logvar - torch.log(self.prior_std**2) + 
            (self.prior_std**2 + self.weight_mu**2) / torch.exp(self.weight_logvar) - 1
        )
        
        # KL divergence for bias
        bias_kl = 0.5 * torch.sum(
            self.bias_logvar - torch.log(self.prior_std**2) + 
            (self.prior_std**2 + self.bias_mu**2) / torch.exp(self.bias_logvar) - 1
        )
        
        return weight_kl + bias_kl


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network for uncertainty quantification.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int = 1,
                 prior_std: float = 1.0,
                 dropout: float = 0.1):
        """
        Initialize Bayesian Neural Network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            prior_std: Prior standard deviation
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_std))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(BayesianLinear(prev_dim, output_dim, prior_std))
        
        self.network = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass through Bayesian Neural Network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            sample: Whether to sample from posterior
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                x = layer(x, sample)
            else:
                x = layer(x)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate total KL divergence.
        
        Returns:
            Total KL divergence value
        """
        total_kl = 0.0
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                total_kl += layer.kl_divergence()
        
        return total_kl
    
    def predict_with_uncertainty(self, 
                                x: torch.Tensor, 
                                num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        self.eval()
        with torch.no_grad():
            predictions = []
            
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
            
            # Calculate mean and uncertainty
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)
            
            return mean_pred, uncertainty


class SpectralNormalizedGaussianProcess(nn.Module):
    """
    Spectral-normalized Neural Gaussian Process for uncertainty quantification.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int = 1,
                 spectral_norm_bound: float = 1.0,
                 dropout: float = 0.1):
        """
        Initialize Spectral-normalized Neural Gaussian Process.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            spectral_norm_bound: Spectral normalization bound
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.spectral_norm_bound = spectral_norm_bound
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Apply spectral normalization
        self.feature_extractor = self._apply_spectral_norm(self.feature_extractor)
        
        # Output layers for mean and variance
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.var_layer = nn.Linear(hidden_dim, output_dim)
        
        # Apply spectral normalization to output layers
        self.mean_layer = nn.utils.spectral_norm(self.mean_layer)
        self.var_layer = nn.utils.spectral_norm(self.var_layer)
    
    def _apply_spectral_norm(self, module: nn.Module) -> nn.Module:
        """Apply spectral normalization to all linear layers."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, nn.utils.spectral_norm(child))
            else:
                self._apply_spectral_norm(child)
        return module
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Spectral-normalized Neural Gaussian Process.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (mean, variance)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict mean and variance
        mean = self.mean_layer(features)
        var = F.softplus(self.var_layer(features)) + 1e-6  # Ensure positive variance
        
        return mean, var
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(x)
            uncertainty = torch.sqrt(var)
            return mean, uncertainty


class UncertaintyQuantificationHead(nn.Module):
    """
    Uncertainty quantification head for fraud detection models.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 method: str = 'bnn',
                 num_samples: int = 100):
        """
        Initialize uncertainty quantification head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            method: Method for uncertainty quantification ('bnn' or 'sngp')
            num_samples: Number of Monte Carlo samples for BNN
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.method = method
        self.num_samples = num_samples
        
        if method == 'bnn':
            self.uncertainty_model = BayesianNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=[hidden_dim, hidden_dim // 2],
                output_dim=output_dim,
                prior_std=1.0,
                dropout=0.1
            )
        elif method == 'sngp':
            self.uncertainty_model = SpectralNormalizedGaussianProcess(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                spectral_norm_bound=1.0,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unknown uncertainty quantification method: {method}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through uncertainty quantification head.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary containing predictions and uncertainty
        """
        if self.method == 'bnn':
            # Bayesian Neural Network
            if self.training:
                predictions = self.uncertainty_model(x, sample=True)
                return {
                    'predictions': predictions,
                    'uncertainty': torch.zeros_like(predictions)  # Placeholder for training
                }
            else:
                mean_pred, uncertainty = self.uncertainty_model.predict_with_uncertainty(
                    x, self.num_samples
                )
                return {
                    'predictions': mean_pred,
                    'uncertainty': uncertainty
                }
        
        elif self.method == 'sngp':
            # Spectral-normalized Neural Gaussian Process
            mean, var = self.uncertainty_model(x)
            uncertainty = torch.sqrt(var)
            return {
                'predictions': mean,
                'uncertainty': uncertainty
            }
    
    def get_kl_divergence(self) -> torch.Tensor:
        """Get KL divergence for BNN (only applicable for BNN method)."""
        if self.method == 'bnn':
            return self.uncertainty_model.kl_divergence()
        else:
            return torch.tensor(0.0)


class ConfidenceCalibrator(nn.Module):
    """
    Confidence calibration for uncertainty estimates.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64):
        """
        Initialize confidence calibrator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.calibrator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through confidence calibrator.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Calibrated confidence scores [batch_size, 1]
        """
        return self.calibrator(x)


if __name__ == "__main__":
    # Test uncertainty quantification models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample data
    batch_size = 32
    input_dim = 128
    
    x = torch.randn(batch_size, input_dim)
    
    # Test Bayesian Neural Network
    bnn = BayesianNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=1,
        prior_std=1.0
    )
    
    bnn_output = bnn(x)
    bnn_kl = bnn.kl_divergence()
    
    print(f"BNN output shape: {bnn_output.shape}")
    print(f"BNN KL divergence: {bnn_kl.item():.4f}")
    
    # Test predictions with uncertainty
    mean_pred, uncertainty = bnn.predict_with_uncertainty(x, num_samples=50)
    print(f"BNN mean predictions shape: {mean_pred.shape}")
    print(f"BNN uncertainty shape: {uncertainty.shape}")
    
    # Test Spectral-normalized Neural Gaussian Process
    sngp = SpectralNormalizedGaussianProcess(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=1
    )
    
    mean, var = sngp(x)
    print(f"SNGP mean shape: {mean.shape}")
    print(f"SNGP variance shape: {var.shape}")
    
    # Test uncertainty quantification head
    uq_head = UncertaintyQuantificationHead(
        input_dim=input_dim,
        hidden_dim=128,
        method='bnn',
        num_samples=50
    )
    
    uq_output = uq_head(x)
    print(f"UQ head output keys: {uq_output.keys()}")
    print(f"UQ head predictions shape: {uq_output['predictions'].shape}")
    print(f"UQ head uncertainty shape: {uq_output['uncertainty'].shape}")
    
    # Test confidence calibrator
    calibrator = ConfidenceCalibrator(input_dim, 64)
    confidence = calibrator(x)
    print(f"Confidence shape: {confidence.shape}")
    print(f"Confidence range: [{confidence.min().item():.4f}, {confidence.max().item():.4f}]")
