"""
Integrated model combining TGNN, Transformer, and HAR-BACD-V for dark pool fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

from .temporal_gnn import TemporalGraphNeuralNetwork
from .transformer_model import TradingTransformer, TemporalTransformer
from .hybrid_model import HARBACDModel
from .uncertainty_quantification import UncertaintyQuantificationHead
from .explainability import ModelExplainer

logger = logging.getLogger(__name__)


class IntegratedFraudDetectionModel(nn.Module):
    """
    Integrated model combining all components for comprehensive fraud detection.
    """
    
    def __init__(self, 
                 config: Dict,
                 feature_names: List[str] = None):
        """
        Initialize integrated fraud detection model.
        
        Args:
            config: Configuration dictionary
            feature_names: List of feature names for explainability
        """
        super().__init__()
        
        self.config = config
        self.feature_names = feature_names or [f"feature_{i}" for i in range(118)]
        
        # Extract configuration
        model_config = config['model']
        tgnn_config = model_config['tgnn']
        transformer_config = model_config['transformer']
        hybrid_config = model_config['hybrid']
        uncertainty_config = model_config['uncertainty']
        
        # Input dimensions
        self.input_dim = config['data']['num_features']
        self.sequence_length = config['data']['sequence_length']
        
        # Temporal Graph Neural Network
        self.tgnn = TemporalGraphNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dim=tgnn_config['hidden_dim'],
            num_layers=tgnn_config['num_layers'],
            num_heads=tgnn_config['num_heads'],
            dropout=tgnn_config['dropout'],
            use_memory=tgnn_config.get('use_memory', True),
            memory_size=tgnn_config.get('memory_size', 1000)
        )
        
        # Transformer for sequential processing
        self.transformer = TradingTransformer(
            input_dim=self.input_dim,
            d_model=transformer_config['d_model'],
            num_heads=transformer_config['num_heads'],
            num_layers=transformer_config['num_layers'],
            d_ff=transformer_config['d_model'] * 4,
            max_seq_length=transformer_config['max_seq_length'],
            dropout=transformer_config['dropout'],
            num_classes=1
        )
        
        # Temporal Transformer for time-series processing
        self.temporal_transformer = TemporalTransformer(
            input_dim=self.input_dim,
            d_model=transformer_config['d_model'],
            num_heads=transformer_config['num_heads'],
            num_layers=transformer_config['num_layers'],
            dropout=transformer_config['dropout']
        )
        
        # HAR-BACD-V Hybrid Model
        self.hybrid_model = HARBACDModel(
            input_dim=self.input_dim,
            hidden_dim=hybrid_config.get('hidden_dim', 128),
            har_lags=hybrid_config['har_lags'],
            bacd_components=hybrid_config['bacd_components'],
            attention_heads=hybrid_config['attention_heads'],
            dropout=hybrid_config.get('dropout', 0.1),
            num_classes=1
        )
        
        # Feature fusion layer
        fusion_dim = (
            tgnn_config['hidden_dim'] +  # TGNN output
            transformer_config['d_model'] +  # Transformer output
            transformer_config['d_model'] +  # Temporal transformer output
            hybrid_config.get('hidden_dim', 128) * 2  # HAR-BACD output
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 4, fusion_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 8, 1)
        )
        
        # Uncertainty quantification head
        if uncertainty_config.get('use_bnn', True):
            self.uncertainty_head = UncertaintyQuantificationHead(
                input_dim=fusion_dim // 4,
                hidden_dim=uncertainty_config.get('hidden_dim', 128),
                output_dim=1,
                method='bnn',
                num_samples=uncertainty_config.get('bnn_samples', 100)
            )
        else:
            self.uncertainty_head = UncertaintyQuantificationHead(
                input_dim=fusion_dim // 4,
                hidden_dim=uncertainty_config.get('hidden_dim', 128),
                output_dim=1,
                method='sngp',
                num_samples=uncertainty_config.get('bnn_samples', 100)
            )
        
        # Information asymmetry score head
        self.asymmetry_head = nn.Sequential(
            nn.Linear(fusion_dim // 4, fusion_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # Confidence calibration
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(fusion_dim // 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through integrated model.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim] or [num_nodes, input_dim]
            edge_index: Edge indices for graph data [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Dictionary containing all model outputs
        """
        # Ensure input is 3D for sequential processing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        batch_size, seq_len, input_dim = x.size()
        
        # 1. Temporal Graph Neural Network
        if edge_index is not None:
            # Reshape for TGNN (flatten sequence dimension)
            x_flat = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
            
            # Create batch assignment for flattened data
            if batch is None:
                batch = torch.arange(batch_size, device=x.device).repeat_interleave(seq_len)
            
            tgnn_output = self.tgnn(x_flat, edge_index, batch)
            tgnn_features = tgnn_output.view(batch_size, -1)  # [batch_size, tgnn_hidden_dim]
        else:
            # Use mean pooling if no graph structure
            tgnn_features = x.mean(dim=1)  # [batch_size, input_dim]
            tgnn_features = F.linear(tgnn_features, 
                                   torch.randn(self.tgnn.hidden_dim, input_dim, device=x.device))
        
        # 2. Transformer for sequential processing
        transformer_output = self.transformer(x)  # [batch_size, 1]
        transformer_features = transformer_output  # [batch_size, 1]
        
        # 3. Temporal Transformer
        temporal_output = self.temporal_transformer(x)  # [batch_size, seq_len, input_dim]
        temporal_features = temporal_output.mean(dim=1)  # [batch_size, input_dim]
        
        # 4. HAR-BACD-V Hybrid Model
        hybrid_outputs = self.hybrid_model(x)
        har_features = hybrid_outputs['har_features']  # [batch_size, hidden_dim]
        bacd_features = hybrid_outputs['bacd_features']  # [batch_size, hidden_dim]
        
        # 5. Feature Fusion
        fused_features = torch.cat([
            tgnn_features,
            transformer_features,
            temporal_features,
            har_features,
            bacd_features
        ], dim=-1)  # [batch_size, fusion_dim]
        
        fused_features = self.feature_fusion(fused_features)  # [batch_size, fusion_dim//4]
        
        # 6. Main Classification
        fraud_logits = self.classifier(fused_features)  # [batch_size, 1]
        
        # 7. Uncertainty Quantification
        uncertainty_output = self.uncertainty_head(fused_features)
        
        # 8. Information Asymmetry Score
        asymmetry_score = self.asymmetry_head(fused_features)  # [batch_size, 1]
        
        # 9. Confidence Calibration
        confidence = self.confidence_calibrator(fused_features)  # [batch_size, 1]
        
        return {
            'fraud_logits': fraud_logits,
            'fraud_probability': torch.sigmoid(fraud_logits),
            'asymmetry_score': asymmetry_score,
            'confidence': confidence,
            'uncertainty': uncertainty_output.get('uncertainty', torch.zeros_like(fraud_logits)),
            'tgnn_features': tgnn_features,
            'transformer_features': transformer_features,
            'temporal_features': temporal_features,
            'har_features': har_features,
            'bacd_features': bacd_features,
            'fused_features': fused_features,
            'hybrid_outputs': hybrid_outputs
        }
    
    def predict_with_uncertainty(self, 
                                x: torch.Tensor,
                                edge_index: Optional[torch.Tensor] = None,
                                batch: Optional[torch.Tensor] = None,
                                num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input features
            edge_index: Edge indices for graph data
            batch: Batch assignment for nodes
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing predictions and uncertainty
        """
        self.eval()
        
        predictions = []
        asymmetry_scores = []
        confidences = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.forward(x, edge_index, batch)
                predictions.append(outputs['fraud_probability'])
                asymmetry_scores.append(outputs['asymmetry_score'])
                confidences.append(outputs['confidence'])
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, 1]
        asymmetry_scores = torch.stack(asymmetry_scores, dim=0)
        confidences = torch.stack(confidences, dim=0)
        
        # Calculate statistics
        mean_predictions = predictions.mean(dim=0)
        std_predictions = predictions.std(dim=0)
        
        mean_asymmetry = asymmetry_scores.mean(dim=0)
        std_asymmetry = asymmetry_scores.std(dim=0)
        
        mean_confidence = confidences.mean(dim=0)
        std_confidence = confidences.std(dim=0)
        
        return {
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions,
            'mean_asymmetry': mean_asymmetry,
            'std_asymmetry': std_asymmetry,
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'all_predictions': predictions,
            'all_asymmetry': asymmetry_scores,
            'all_confidences': confidences
        }
    
    def get_explainer(self, device: str = 'cpu') -> ModelExplainer:
        """
        Get model explainer for interpretability.
        
        Args:
            device: Device to run on
            
        Returns:
            ModelExplainer instance
        """
        return ModelExplainer(
            model=self,
            feature_names=self.feature_names,
            class_names=['Normal', 'Fraud'],
            device=device
        )
    
    def get_model_summary(self) -> Dict[str, int]:
        """
        Get model summary with parameter counts.
        
        Returns:
            Dictionary containing parameter counts for each component
        """
        return {
            'tgnn_parameters': sum(p.numel() for p in self.tgnn.parameters()),
            'transformer_parameters': sum(p.numel() for p in self.transformer.parameters()),
            'temporal_transformer_parameters': sum(p.numel() for p in self.temporal_transformer.parameters()),
            'hybrid_model_parameters': sum(p.numel() for p in self.hybrid_model.parameters()),
            'fusion_parameters': sum(p.numel() for p in self.feature_fusion.parameters()),
            'classifier_parameters': sum(p.numel() for p in self.classifier.parameters()),
            'uncertainty_parameters': sum(p.numel() for p in self.uncertainty_head.parameters()),
            'asymmetry_parameters': sum(p.numel() for p in self.asymmetry_head.parameters()),
            'total_parameters': sum(p.numel() for p in self.parameters())
        }


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple integrated models for improved performance.
    """
    
    def __init__(self, 
                 models: List[IntegratedFraudDetectionModel],
                 weights: Optional[List[float]] = None):
        """
        Initialize model ensemble.
        
        Args:
            models: List of integrated models
            weights: Weights for each model (if None, equal weights)
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            self.weights = weights
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features
            edge_index: Edge indices for graph data
            batch: Batch assignment for nodes
            
        Returns:
            Dictionary containing ensemble predictions
        """
        all_outputs = []
        
        for model in self.models:
            outputs = model(x, edge_index, batch)
            all_outputs.append(outputs)
        
        # Weighted average of predictions
        ensemble_outputs = {}
        
        for key in all_outputs[0].keys():
            if key in ['fraud_logits', 'fraud_probability', 'asymmetry_score', 'confidence']:
                weighted_sum = sum(
                    self.weights[i] * all_outputs[i][key] 
                    for i in range(self.num_models)
                )
                ensemble_outputs[key] = weighted_sum
            else:
                # For other outputs, just take the first model's output
                ensemble_outputs[key] = all_outputs[0][key]
        
        return ensemble_outputs


if __name__ == "__main__":
    # Test the integrated model
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = IntegratedFraudDetectionModel(
        config=config,
        feature_names=[f"feature_{i}" for i in range(118)]
    )
    
    # Create sample data
    batch_size = 32
    seq_len = 100
    input_dim = 118
    
    x = torch.randn(batch_size, seq_len, input_dim)
    edge_index = torch.randint(0, batch_size * seq_len, (2, 200))
    batch = torch.arange(batch_size).repeat_interleave(seq_len)
    
    # Forward pass
    outputs = model(x, edge_index, batch)
    
    print("Integrated Model Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Test uncertainty quantification
    uncertainty_outputs = model.predict_with_uncertainty(x, edge_index, batch, num_samples=10)
    
    print("\nUncertainty Quantification:")
    for key, value in uncertainty_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # Model summary
    summary = model.get_model_summary()
    print(f"\nModel Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:,}")
    
    print(f"Total parameters: {summary['total_parameters']:,}")
