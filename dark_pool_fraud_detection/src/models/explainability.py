"""
Explainable AI (XAI) module for model interpretability using SHAP and LIME.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import SHAP and LIME (these will be installed via requirements.txt)
try:
    import shap
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    SHAP_AVAILABLE = True
    LIME_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    LIME_AVAILABLE = False
    logging.warning("SHAP and/or LIME not available. Install with: pip install shap lime")

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Main class for model explainability using SHAP and LIME.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 feature_names: List[str],
                 class_names: List[str] = None,
                 device: str = 'cpu'):
        """
        Initialize model explainer.
        
        Args:
            model: Trained PyTorch model
            feature_names: List of feature names
            class_names: List of class names
            device: Device to run on
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Normal', 'Fraud']
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Cache for explanations
        self.explanation_cache = {}
    
    def setup_shap_explainer(self, 
                            background_data: torch.Tensor,
                            explainer_type: str = 'kernel') -> None:
        """
        Setup SHAP explainer.
        
        Args:
            background_data: Background data for SHAP explainer
            explainer_type: Type of SHAP explainer ('kernel', 'deep', 'gradient')
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available. Install with: pip install shap")
            return
        
        try:
            # Convert background data to numpy
            background_np = background_data.cpu().numpy()
            
            if explainer_type == 'kernel':
                self.shap_explainer = shap.KernelExplainer(
                    self._model_predict_wrapper,
                    background_np
                )
            elif explainer_type == 'deep':
                self.shap_explainer = shap.DeepExplainer(
                    self.model,
                    background_data
                )
            elif explainer_type == 'gradient':
                self.shap_explainer = shap.GradientExplainer(
                    self.model,
                    background_data
                )
            else:
                raise ValueError(f"Unknown SHAP explainer type: {explainer_type}")
            
            logger.info(f"SHAP {explainer_type} explainer setup successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup SHAP explainer: {e}")
            self.shap_explainer = None
    
    def setup_lime_explainer(self, 
                            training_data: torch.Tensor,
                            mode: str = 'classification') -> None:
        """
        Setup LIME explainer.
        
        Args:
            training_data: Training data for LIME explainer
            mode: Mode of LIME explainer ('classification' or 'regression')
        """
        if not LIME_AVAILABLE:
            logger.error("LIME not available. Install with: pip install lime")
            return
        
        try:
            # Convert training data to numpy
            training_np = training_data.cpu().numpy()
            
            self.lime_explainer = LimeTabularExplainer(
                training_np,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=mode,
                discretize_continuous=True
            )
            
            logger.info("LIME explainer setup successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LIME explainer: {e}")
            self.lime_explainer = None
    
    def _model_predict_wrapper(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper function for model prediction (for SHAP).
        
        Args:
            x: Input data [batch_size, num_features]
            
        Returns:
            Model predictions [batch_size, num_classes]
        """
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            predictions = self.model(x_tensor)
            
            # Convert to probabilities if needed
            if predictions.dim() > 1 and predictions.size(1) > 1:
                predictions = torch.softmax(predictions, dim=1)
            
            return predictions.cpu().numpy()
    
    def explain_shap(self, 
                    x: torch.Tensor,
                    max_evals: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate SHAP explanations.
        
        Args:
            x: Input data to explain [batch_size, num_features]
            max_evals: Maximum number of evaluations for SHAP
            
        Returns:
            Dictionary containing SHAP values and other information
        """
        if self.shap_explainer is None:
            logger.error("SHAP explainer not setup. Call setup_shap_explainer first.")
            return {}
        
        try:
            # Convert to numpy
            x_np = x.cpu().numpy()
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(
                x_np,
                max_evals=max_evals
            )
            
            # Handle different SHAP explainer outputs
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values = shap_values[1]  # Use fraud class
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Get top features
            top_features_idx = np.argsort(feature_importance)[::-1]
            top_features = {
                'indices': top_features_idx,
                'names': [self.feature_names[i] for i in top_features_idx],
                'importance': feature_importance[top_features_idx]
            }
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'top_features': top_features,
                'base_value': self.shap_explainer.expected_value
            }
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            return {}
    
    def explain_lime(self, 
                    x: torch.Tensor,
                    num_features: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate LIME explanations.
        
        Args:
            x: Input data to explain [batch_size, num_features]
            num_features: Number of top features to explain
            
        Returns:
            Dictionary containing LIME explanations
        """
        if self.lime_explainer is None:
            logger.error("LIME explainer not setup. Call setup_lime_explainer first.")
            return {}
        
        try:
            # Convert to numpy
            x_np = x.cpu().numpy()
            
            explanations = []
            feature_importance = np.zeros((x.size(0), len(self.feature_names)))
            
            # Generate explanations for each sample
            for i in range(x.size(0)):
                explanation = self.lime_explainer.explain_instance(
                    x_np[i],
                    self._model_predict_wrapper,
                    num_features=num_features
                )
                
                explanations.append(explanation)
                
                # Extract feature importance
                for feature_idx, importance in explanation.as_list():
                    if isinstance(feature_idx, str):
                        # Feature name to index
                        try:
                            feature_idx = self.feature_names.index(feature_idx)
                        except ValueError:
                            continue
                    
                    feature_importance[i, feature_idx] = importance
            
            # Calculate average feature importance
            avg_feature_importance = np.abs(feature_importance).mean(axis=0)
            
            # Get top features
            top_features_idx = np.argsort(avg_feature_importance)[::-1]
            top_features = {
                'indices': top_features_idx,
                'names': [self.feature_names[i] for i in top_features_idx],
                'importance': avg_feature_importance[top_features_idx]
            }
            
            return {
                'explanations': explanations,
                'feature_importance': feature_importance,
                'avg_feature_importance': avg_feature_importance,
                'top_features': top_features
            }
            
        except Exception as e:
            logger.error(f"Failed to generate LIME explanations: {e}")
            return {}
    
    def explain_batch(self, 
                     x: torch.Tensor,
                     methods: List[str] = ['shap', 'lime'],
                     **kwargs) -> Dict[str, Dict]:
        """
        Generate explanations using multiple methods.
        
        Args:
            x: Input data to explain [batch_size, num_features]
            methods: List of explanation methods to use
            **kwargs: Additional arguments for explanation methods
            
        Returns:
            Dictionary containing explanations from all methods
        """
        explanations = {}
        
        if 'shap' in methods and self.shap_explainer is not None:
            explanations['shap'] = self.explain_shap(x, **kwargs)
        
        if 'lime' in methods and self.lime_explainer is not None:
            explanations['lime'] = self.explain_lime(x, **kwargs)
        
        return explanations
    
    def get_feature_importance_summary(self, 
                                     x: torch.Tensor,
                                     methods: List[str] = ['shap', 'lime']) -> Dict[str, np.ndarray]:
        """
        Get feature importance summary from multiple methods.
        
        Args:
            x: Input data to explain [batch_size, num_features]
            methods: List of explanation methods to use
            
        Returns:
            Dictionary containing feature importance from all methods
        """
        explanations = self.explain_batch(x, methods)
        
        feature_importance_summary = {}
        
        for method, explanation in explanations.items():
            if 'feature_importance' in explanation:
                feature_importance_summary[method] = explanation['feature_importance']
            elif 'avg_feature_importance' in explanation:
                feature_importance_summary[method] = explanation['avg_feature_importance']
        
        return feature_importance_summary
    
    def visualize_explanations(self, 
                              x: torch.Tensor,
                              sample_idx: int = 0,
                              methods: List[str] = ['shap', 'lime'],
                              top_k: int = 10) -> None:
        """
        Visualize explanations for a single sample.
        
        Args:
            x: Input data to explain [batch_size, num_features]
            sample_idx: Index of sample to visualize
            methods: List of explanation methods to use
            top_k: Number of top features to show
        """
        if sample_idx >= x.size(0):
            logger.error(f"Sample index {sample_idx} out of range")
            return
        
        explanations = self.explain_batch(x, methods)
        
        for method, explanation in explanations.items():
            if method == 'shap' and 'top_features' in explanation:
                top_features = explanation['top_features']
                print(f"\nSHAP Top {top_k} Features:")
                for i in range(min(top_k, len(top_features['names']))):
                    print(f"{i+1}. {top_features['names'][i]}: {top_features['importance'][i]:.4f}")
            
            elif method == 'lime' and 'top_features' in explanation:
                top_features = explanation['top_features']
                print(f"\nLIME Top {top_k} Features:")
                for i in range(min(top_k, len(top_features['names']))):
                    print(f"{i+1}. {top_features['names'][i]}: {top_features['importance'][i]:.4f}")


class AttentionVisualizer:
    """
    Visualizer for attention weights in transformer models.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: Transformer model with attention layers
        """
        self.model = model
        self.attention_weights = []
        
        # Register hooks to capture attention weights
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                self.attention_weights.append(module.attention_weights)
        
        # Register hooks for attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                module.register_forward_hook(hook_fn)
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights for input.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weight tensors
        """
        self.attention_weights = []
        
        with torch.no_grad():
            _ = self.model(x)
        
        return self.attention_weights
    
    def visualize_attention(self, 
                           x: torch.Tensor,
                           layer_idx: int = 0,
                           head_idx: int = 0) -> None:
        """
        Visualize attention weights.
        
        Args:
            x: Input tensor
            layer_idx: Index of attention layer to visualize
            head_idx: Index of attention head to visualize
        """
        attention_weights = self.get_attention_weights(x)
        
        if not attention_weights:
            print("No attention weights found")
            return
        
        if layer_idx >= len(attention_weights):
            print(f"Layer index {layer_idx} out of range")
            return
        
        weights = attention_weights[layer_idx]
        
        if weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
            weights = weights[0, head_idx]  # [seq_len, seq_len]
        elif weights.dim() == 3:  # [batch, seq_len, seq_len]
            weights = weights[0]  # [seq_len, seq_len]
        
        print(f"Attention weights shape: {weights.shape}")
        print(f"Attention weights for layer {layer_idx}, head {head_idx}:")
        print(weights.cpu().numpy())


if __name__ == "__main__":
    # Test the explainability module
    import torch.nn as nn
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create model and explainer
    model = SimpleModel(64, 128, 1)
    feature_names = [f"feature_{i}" for i in range(64)]
    
    explainer = ModelExplainer(
        model=model,
        feature_names=feature_names,
        class_names=['Normal', 'Fraud']
    )
    
    # Create sample data
    x = torch.randn(10, 64)
    background_data = torch.randn(100, 64)
    training_data = torch.randn(1000, 64)
    
    # Setup explainers
    explainer.setup_shap_explainer(background_data, 'kernel')
    explainer.setup_lime_explainer(training_data, 'classification')
    
    # Generate explanations
    explanations = explainer.explain_batch(x, methods=['shap', 'lime'])
    
    print("Explanations generated successfully!")
    print(f"Available methods: {list(explanations.keys())}")
    
    # Visualize explanations
    explainer.visualize_explanations(x, sample_idx=0, methods=['shap', 'lime'], top_k=5)
