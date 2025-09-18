"""
Real-time inference system for dark pool fraud detection.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from pathlib import Path
import yaml
import json
from collections import deque
import threading
import queue
from dataclasses import dataclass

from ..models.integrated_model import IntegratedFraudDetectionModel
from ..models.explainability import ModelExplainer
from ..data_pipeline.temporal_graph import TemporalGraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class FraudPrediction:
    """Data class for fraud prediction results."""
    timestamp: float
    fraud_probability: float
    asymmetry_score: float
    confidence: float
    uncertainty: float
    is_fraud: bool
    explanation: Optional[Dict] = None
    processing_time_ms: float = 0.0


class RealTimeInference:
    """
    Real-time inference system for fraud detection.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 max_latency_ms: float = 2.3,
                 batch_size: int = 1,
                 enable_explanations: bool = True,
                 enable_uncertainty: bool = True):
        """
        Initialize real-time inference system.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            max_latency_ms: Maximum allowed latency in milliseconds
            batch_size: Batch size for inference
            enable_explanations: Whether to enable model explanations
            enable_uncertainty: Whether to enable uncertainty quantification
        """
        self.model_path = model_path
        self.config_path = config_path
        self.max_latency_ms = max_latency_ms
        self.batch_size = batch_size
        self.enable_explanations = enable_explanations
        self.enable_uncertainty = enable_uncertainty
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Initialize temporal graph builder
        self.temporal_graph_builder = TemporalGraphBuilder(
            self.config.get('temporal_graph', {})
        )
        
        # Initialize explainer if enabled
        self.explainer = None
        if self.enable_explanations:
            self.explainer = self._setup_explainer()
        
        # Inference queue and thread
        self.inference_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.inference_thread = None
        self.running = False
        
        # Performance tracking
        self.latency_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Model warmup
        self._warmup_model()
        
        logger.info("Real-time inference system initialized")
    
    def _load_model(self) -> IntegratedFraudDetectionModel:
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Create model
        model = IntegratedFraudDetectionModel(
            config=self.config,
            feature_names=[f"feature_{i}" for i in range(self.config['data']['num_features'])]
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        model.to(self.device)
        
        logger.info("Model loaded successfully")
        return model
    
    def _setup_explainer(self) -> ModelExplainer:
        """Setup model explainer."""
        try:
            explainer = ModelExplainer(
                model=self.model,
                feature_names=[f"feature_{i}" for i in range(self.config['data']['num_features'])],
                class_names=['Normal', 'Fraud'],
                device=self.device
            )
            
            # Setup SHAP explainer with background data
            background_data = torch.randn(100, self.config['data']['num_features']).to(self.device)
            explainer.setup_shap_explainer(background_data, 'kernel')
            
            logger.info("Model explainer setup successfully")
            return explainer
            
        except Exception as e:
            logger.warning(f"Failed to setup explainer: {e}")
            return None
    
    def _warmup_model(self):
        """Warmup the model with dummy data."""
        logger.info("Warming up model...")
        
        # Create dummy data
        dummy_data = torch.randn(self.batch_size, self.config['data']['sequence_length'], 
                               self.config['data']['num_features']).to(self.device)
        
        # Run inference
        with torch.no_grad():
            _ = self.model(dummy_data)
        
        logger.info("Model warmup completed")
    
    def start_inference_thread(self):
        """Start the inference thread."""
        if self.inference_thread is not None and self.inference_thread.is_alive():
            logger.warning("Inference thread already running")
            return
        
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        logger.info("Inference thread started")
    
    def stop_inference_thread(self):
        """Stop the inference thread."""
        self.running = False
        if self.inference_thread is not None:
            self.inference_thread.join()
        
        logger.info("Inference thread stopped")
    
    def _inference_loop(self):
        """Main inference loop running in separate thread."""
        while self.running:
            try:
                # Get batch from queue
                batch_data = self.inference_queue.get(timeout=0.1)
                
                # Process batch
                results = self._process_batch(batch_data)
                
                # Put results in result queue
                self.result_queue.put(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
    
    def _process_batch(self, batch_data: Dict) -> List[FraudPrediction]:
        """Process a batch of data."""
        start_time = time.time()
        
        # Extract data
        features = batch_data['features'].to(self.device)
        timestamps = batch_data.get('timestamps', [time.time()] * features.size(0))
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(features)
        
        # Process results
        results = []
        for i in range(features.size(0)):
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create prediction
            prediction = FraudPrediction(
                timestamp=timestamps[i],
                fraud_probability=outputs['fraud_probability'][i].item(),
                asymmetry_score=outputs['asymmetry_score'][i].item(),
                confidence=outputs['confidence'][i].item(),
                uncertainty=outputs['uncertainty'][i].item() if 'uncertainty' in outputs else 0.0,
                is_fraud=outputs['fraud_probability'][i].item() > 0.5,
                processing_time_ms=processing_time
            )
            
            # Add explanation if enabled
            if self.enable_explanations and self.explainer is not None:
                try:
                    explanation = self.explainer.explain_shap(features[i:i+1])
                    prediction.explanation = explanation
                except Exception as e:
                    logger.warning(f"Failed to generate explanation: {e}")
            
            results.append(prediction)
        
        # Track performance
        self.latency_history.append(processing_time)
        
        return results
    
    def predict(self, 
                features: np.ndarray,
                timestamps: Optional[List[float]] = None,
                blocking: bool = True) -> List[FraudPrediction]:
        """
        Make predictions on input features.
        
        Args:
            features: Input features [batch_size, seq_len, num_features]
            timestamps: List of timestamps for each sample
            blocking: Whether to wait for results
            
        Returns:
            List of fraud predictions
        """
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Ensure correct shape
        if features.dim() == 2:
            features = features.unsqueeze(1)  # Add sequence dimension
        
        # Set timestamps
        if timestamps is None:
            timestamps = [time.time()] * features.size(0)
        
        # Create batch data
        batch_data = {
            'features': features,
            'timestamps': timestamps
        }
        
        if blocking:
            # Process immediately
            return self._process_batch(batch_data)
        else:
            # Add to queue
            self.inference_queue.put(batch_data)
            
            # Get results
            try:
                results = self.result_queue.get(timeout=1.0)
                return results
            except queue.Empty:
                logger.warning("Inference timeout")
                return []
    
    def predict_single(self, 
                      features: np.ndarray,
                      timestamp: Optional[float] = None) -> FraudPrediction:
        """
        Make prediction on a single sample.
        
        Args:
            features: Input features [seq_len, num_features]
            timestamp: Timestamp for the sample
            
        Returns:
            Fraud prediction
        """
        # Add batch dimension
        features = features.unsqueeze(0)
        
        # Set timestamp
        if timestamp is None:
            timestamp = time.time()
        
        # Make prediction
        results = self.predict(features, [timestamp], blocking=True)
        
        return results[0]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.latency_history:
            return {}
        
        latencies = list(self.latency_history)
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'std_latency_ms': np.std(latencies),
            'total_predictions': len(latencies)
        }
    
    def is_performance_acceptable(self) -> bool:
        """Check if current performance meets requirements."""
        stats = self.get_performance_stats()
        
        if not stats:
            return False
        
        # Check if mean latency is below threshold
        return stats['mean_latency_ms'] <= self.max_latency_ms
    
    def save_performance_log(self, path: str):
        """Save performance log to file."""
        stats = self.get_performance_stats()
        
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Performance log saved to {path}")


class BatchInference:
    """
    Batch inference system for processing large datasets.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 batch_size: int = 32):
        """
        Initialize batch inference system.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.config_path = config_path
        self.batch_size = batch_size
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        logger.info("Batch inference system initialized")
    
    def _load_model(self) -> IntegratedFraudDetectionModel:
        """Load the trained model."""
        model = IntegratedFraudDetectionModel(
            config=self.config,
            feature_names=[f"feature_{i}" for i in range(self.config['data']['num_features'])]
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def predict_batch(self, 
                     features: np.ndarray,
                     timestamps: Optional[List[float]] = None) -> List[FraudPrediction]:
        """
        Make predictions on a batch of data.
        
        Args:
            features: Input features [num_samples, seq_len, num_features]
            timestamps: List of timestamps for each sample
            
        Returns:
            List of fraud predictions
        """
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Set timestamps
        if timestamps is None:
            timestamps = [time.time()] * features.size(0)
        
        # Process in batches
        all_results = []
        
        for i in range(0, features.size(0), self.batch_size):
            batch_features = features[i:i+self.batch_size]
            batch_timestamps = timestamps[i:i+self.batch_size]
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_features)
            
            # Process results
            for j in range(batch_features.size(0)):
                prediction = FraudPrediction(
                    timestamp=batch_timestamps[j],
                    fraud_probability=outputs['fraud_probability'][j].item(),
                    asymmetry_score=outputs['asymmetry_score'][j].item(),
                    confidence=outputs['confidence'][j].item(),
                    uncertainty=outputs['uncertainty'][j].item() if 'uncertainty' in outputs else 0.0,
                    is_fraud=outputs['fraud_probability'][j].item() > 0.5,
                    processing_time_ms=0.0  # Not tracked for batch inference
                )
                
                all_results.append(prediction)
        
        return all_results


if __name__ == "__main__":
    # Test the inference system
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create real-time inference system
    inference = RealTimeInference(
        model_path='models/best_model.pth',
        config_path='configs/config.yaml',
        max_latency_ms=2.3,
        enable_explanations=True
    )
    
    # Start inference thread
    inference.start_inference_thread()
    
    # Create sample data
    features = np.random.randn(10, 100, 118)  # 10 samples, 100 timesteps, 118 features
    
    # Make predictions
    results = inference.predict(features, blocking=True)
    
    print(f"Generated {len(results)} predictions")
    for i, result in enumerate(results):
        print(f"Sample {i}: Fraud={result.is_fraud}, Prob={result.fraud_probability:.4f}, "
              f"Confidence={result.confidence:.4f}, Latency={result.processing_time_ms:.2f}ms")
    
    # Get performance stats
    stats = inference.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Stop inference thread
    inference.stop_inference_thread()
