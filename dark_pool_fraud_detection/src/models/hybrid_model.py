"""
Hybrid HAR-BACD-V model for multi-scale analysis of dark pool trading data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HeterogeneousAutoregressive(nn.Module):
    """
    Heterogeneous Autoregressive (HAR) component for macro-level pattern analysis.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 har_lags: List[int] = [1, 5, 22],  # Daily, weekly, monthly
                 dropout: float = 0.1):
        """
        Initialize HAR component.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            har_lags: List of lag periods for HAR
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.har_lags = har_lags
        self.max_lag = max(har_lags)
        
        # HAR layers for different lag periods
        self.har_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
            for _ in range(len(har_lags))
        ])
        
        # Aggregation layer
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim // 2 * len(har_lags), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through HAR component.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            HAR features [batch_size, hidden_dim]
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Ensure we have enough history
        if seq_len < self.max_lag:
            # Pad with zeros if necessary
            padding = self.max_lag - seq_len
            x = F.pad(x, (0, 0, padding, 0), value=0)
            seq_len = self.max_lag
        
        har_outputs = []
        
        # Process each lag period
        for i, lag in enumerate(self.har_lags):
            if lag <= seq_len:
                # Take the last 'lag' timesteps
                lag_data = x[:, -lag:, :]  # [batch_size, lag, input_dim]
                
                # Average over the lag period
                lag_avg = lag_data.mean(dim=1)  # [batch_size, input_dim]
                
                # Process through HAR layer
                har_out = self.har_layers[i](lag_avg)  # [batch_size, hidden_dim//2]
                har_outputs.append(har_out)
            else:
                # Use zeros if not enough history
                har_out = torch.zeros(batch_size, self.hidden_dim // 2, device=x.device)
                har_outputs.append(har_out)
        
        # Concatenate HAR outputs
        har_concat = torch.cat(har_outputs, dim=-1)  # [batch_size, hidden_dim//2 * len(har_lags)]
        
        # Aggregate
        har_features = self.aggregation(har_concat)  # [batch_size, hidden_dim]
        
        return har_features


class BehavioralAutoregressiveConditionalDuration(nn.Module):
    """
    Behavioral Autoregressive Conditional Duration (BACD) component for micro-level analysis.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 bacd_components: int = 3,
                 dropout: float = 0.1):
        """
        Initialize BACD component.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            bacd_components: Number of BACD components
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bacd_components = bacd_components
        
        # Duration modeling layers
        self.duration_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
            for _ in range(bacd_components)
        ])
        
        # Behavioral pattern detection
        self.behavioral_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Aggregation layer
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim // 2 * (bacd_components + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BACD component.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            BACD features [batch_size, hidden_dim]
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Calculate inter-trade durations (time differences)
        if seq_len > 1:
            durations = x[:, 1:, :] - x[:, :-1, :]  # [batch_size, seq_len-1, input_dim]
            # Use the last duration for prediction
            last_duration = durations[:, -1:, :]  # [batch_size, 1, input_dim]
        else:
            last_duration = x  # [batch_size, 1, input_dim]
        
        bacd_outputs = []
        
        # Process each BACD component
        for i in range(self.bacd_components):
            bacd_out = self.duration_layers[i](last_duration.squeeze(1))  # [batch_size, hidden_dim//2]
            bacd_outputs.append(bacd_out)
        
        # Behavioral pattern detection
        behavioral_out = self.behavioral_detector(x[:, -1, :])  # [batch_size, hidden_dim//2]
        bacd_outputs.append(behavioral_out)
        
        # Concatenate BACD outputs
        bacd_concat = torch.cat(bacd_outputs, dim=-1)  # [batch_size, hidden_dim//2 * (bacd_components + 1)]
        
        # Aggregate
        bacd_features = self.aggregation(bacd_concat)  # [batch_size, hidden_dim]
        
        return bacd_features


class DualStageAttention(nn.Module):
    """
    Dual-stage attention mechanism for combining macro and micro analysis.
    """
    
    def __init__(self, 
                 har_dim: int,
                 bacd_dim: int,
                 attention_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize dual-stage attention.
        
        Args:
            har_dim: HAR feature dimension
            bacd_dim: BACD feature dimension
            attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.har_dim = har_dim
        self.bacd_dim = bacd_dim
        self.attention_heads = attention_heads
        
        # Attention layers
        self.har_attention = nn.MultiheadAttention(
            embed_dim=har_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.bacd_attention = nn.MultiheadAttention(
            embed_dim=bacd_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention between HAR and BACD
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=har_dim + bacd_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(har_dim + bacd_dim, har_dim + bacd_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, 
                har_features: torch.Tensor,
                bacd_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual-stage attention.
        
        Args:
            har_features: HAR features [batch_size, har_dim]
            bacd_features: BACD features [batch_size, bacd_dim]
            
        Returns:
            Attended features [batch_size, har_dim + bacd_dim]
        """
        batch_size = har_features.size(0)
        
        # Self-attention for HAR features
        har_attended, _ = self.har_attention(
            har_features.unsqueeze(1),  # [batch_size, 1, har_dim]
            har_features.unsqueeze(1),
            har_features.unsqueeze(1)
        )
        har_attended = har_attended.squeeze(1)  # [batch_size, har_dim]
        
        # Self-attention for BACD features
        bacd_attended, _ = self.bacd_attention(
            bacd_features.unsqueeze(1),  # [batch_size, 1, bacd_dim]
            bacd_features.unsqueeze(1),
            bacd_features.unsqueeze(1)
        )
        bacd_attended = bacd_attended.squeeze(1)  # [batch_size, bacd_dim]
        
        # Cross-attention between HAR and BACD
        combined = torch.cat([har_attended, bacd_attended], dim=-1)  # [batch_size, har_dim + bacd_dim]
        
        cross_attended, _ = self.cross_attention(
            combined.unsqueeze(1),  # [batch_size, 1, har_dim + bacd_dim]
            combined.unsqueeze(1),
            combined.unsqueeze(1)
        )
        cross_attended = cross_attended.squeeze(1)  # [batch_size, har_dim + bacd_dim]
        
        # Output projection
        output = self.output_proj(cross_attended)
        
        return output


class HARBACDModel(nn.Module):
    """
    Hybrid HAR-BACD-V model for dark pool fraud detection.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 har_lags: List[int] = [1, 5, 22],
                 bacd_components: int = 3,
                 attention_heads: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = 1):
        """
        Initialize HAR-BACD-V model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            har_lags: List of lag periods for HAR
            bacd_components: Number of BACD components
            attention_heads: Number of attention heads
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # HAR component
        self.har = HeterogeneousAutoregressive(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            har_lags=har_lags,
            dropout=dropout
        )
        
        # BACD component
        self.bacd = BehavioralAutoregressiveConditionalDuration(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bacd_components=bacd_components,
            dropout=dropout
        )
        
        # Dual-stage attention
        self.attention = DualStageAttention(
            har_dim=hidden_dim,
            bacd_dim=hidden_dim,
            attention_heads=attention_heads,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Information asymmetry score head
        self.asymmetry_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HAR-BACD-V model.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary containing predictions and asymmetry scores
        """
        # HAR component (macro-level analysis)
        har_features = self.har(x)  # [batch_size, hidden_dim]
        
        # BACD component (micro-level analysis)
        bacd_features = self.bacd(x)  # [batch_size, hidden_dim]
        
        # Dual-stage attention
        attended_features = self.attention(har_features, bacd_features)  # [batch_size, hidden_dim * 2]
        
        # Classification
        fraud_logits = self.classifier(attended_features)  # [batch_size, num_classes]
        
        # Information asymmetry score
        asymmetry_score = self.asymmetry_head(attended_features)  # [batch_size, 1]
        
        return {
            'fraud_logits': fraud_logits,
            'asymmetry_score': asymmetry_score,
            'har_features': har_features,
            'bacd_features': bacd_features,
            'attended_features': attended_features
        }


if __name__ == "__main__":
    # Test the HAR-BACD-V model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample data
    batch_size = 32
    seq_len = 100
    input_dim = 64
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create model
    model = HARBACDModel(
        input_dim=input_dim,
        hidden_dim=128,
        har_lags=[1, 5, 22],
        bacd_components=3,
        attention_heads=4,
        dropout=0.1
    )
    
    # Forward pass
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Fraud logits shape: {outputs['fraud_logits'].shape}")
    print(f"Asymmetry score shape: {outputs['asymmetry_score'].shape}")
    print(f"HAR features shape: {outputs['har_features'].shape}")
    print(f"BACD features shape: {outputs['bacd_features'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test individual components
    har = HeterogeneousAutoregressive(input_dim, 128, [1, 5, 22])
    bacd = BehavioralAutoregressiveConditionalDuration(input_dim, 128, 3)
    
    har_out = har(x)
    bacd_out = bacd(x)
    
    print(f"HAR output shape: {har_out.shape}")
    print(f"BACD output shape: {bacd_out.shape}")
