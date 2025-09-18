#!/usr/bin/env python3
"""
Demo script for the dark pool fraud detection system.
"""

import argparse
import logging
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from typing import Dict, List, Any

from src.models.integrated_model import IntegratedFraudDetectionModel
from src.inference.real_time_inference import RealTimeInference, BatchInference
from src.models.explainability import ModelExplainer
from src.data_pipeline.temporal_graph import TemporalGraphBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionDemo:
    """
    Demo class for showcasing the fraud detection system.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str):
        """
        Initialize demo.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        # Initialize inference system
        self.inference = RealTimeInference(
            model_path=model_path,
            config_path=config_path,
            max_latency_ms=2.3,
            enable_explanations=True,
            enable_uncertainty=True
        )
        
        # Initialize explainer
        self.explainer = self._setup_explainer()
        
        logger.info("Demo initialized successfully")
    
    def _load_model(self) -> IntegratedFraudDetectionModel:
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        
        model = IntegratedFraudDetectionModel(
            config=self.config,
            feature_names=[f"feature_{i}" for i in range(self.config['data']['num_features'])]
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        model.to(self.device)
        
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
            
            return explainer
            
        except Exception as e:
            logger.warning(f"Failed to setup explainer: {e}")
            return None
    
    def generate_synthetic_data(self, 
                               num_samples: int = 1000,
                               fraud_ratio: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Generate synthetic dark pool trading data for demo.
        
        Args:
            num_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent samples
            
        Returns:
            Dictionary containing synthetic data
        """
        logger.info(f"Generating {num_samples} synthetic samples with {fraud_ratio:.1%} fraud ratio")
        
        np.random.seed(42)
        
        # Generate base features
        features = np.random.randn(num_samples, self.config['data']['sequence_length'], 
                                 self.config['data']['num_features'])
        
        # Generate labels
        labels = np.random.choice([0, 1], size=num_samples, p=[1-fraud_ratio, fraud_ratio])
        
        # Add fraud patterns to fraudulent samples
        fraud_indices = np.where(labels == 1)[0]
        
        for idx in fraud_indices:
            # Add momentum ignition pattern
            if np.random.random() < 0.3:
                # Sudden price spikes
                spike_start = np.random.randint(0, self.config['data']['sequence_length'] - 10)
                spike_end = spike_start + 10
                features[idx, spike_start:spike_end, 0] += np.random.normal(0, 2, 10)
            
            # Add liquidity fade pattern
            if np.random.random() < 0.4:
                # Decreasing volume over time
                volume_decay = np.linspace(1.0, 0.1, self.config['data']['sequence_length'])
                features[idx, :, 1] *= volume_decay
            
            # Add unusual order sizes
            if np.random.random() < 0.5:
                # Large order sizes
                large_orders = np.random.choice(self.config['data']['sequence_length'], 
                                              size=5, replace=False)
                features[idx, large_orders, 2] *= np.random.uniform(5, 10, 5)
            
            # Add timing anomalies
            if np.random.random() < 0.3:
                # Unusual execution timing
                timing_anomalies = np.random.choice(self.config['data']['sequence_length'], 
                                                  size=3, replace=False)
                features[idx, timing_anomalies, 3] += np.random.normal(0, 3, 3)
        
        # Add noise to all samples
        features += np.random.normal(0, 0.1, features.shape)
        
        # Generate timestamps
        timestamps = np.arange(num_samples) * 60  # 1 minute intervals
        
        return {
            'features': features,
            'labels': labels,
            'timestamps': timestamps
        }
    
    def run_real_time_demo(self, 
                          num_samples: int = 100,
                          delay_ms: int = 100):
        """
        Run real-time inference demo.
        
        Args:
            num_samples: Number of samples to process
            delay_ms: Delay between samples in milliseconds
        """
        logger.info("Starting real-time inference demo")
        
        # Generate synthetic data
        data = self.generate_synthetic_data(num_samples, fraud_ratio=0.1)
        
        # Start inference thread
        self.inference.start_inference_thread()
        
        results = []
        
        print("\n" + "="*80)
        print("REAL-TIME FRAUD DETECTION DEMO")
        print("="*80)
        print(f"Processing {num_samples} samples with {delay_ms}ms delay...")
        print("-"*80)
        
        for i in range(num_samples):
            # Get sample
            sample_features = data['features'][i:i+1]
            sample_label = data['labels'][i]
            sample_timestamp = data['timestamps'][i]
            
            # Make prediction
            start_time = time.time()
            prediction = self.inference.predict_single(
                sample_features[0], 
                sample_timestamp
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Store result
            results.append({
                'sample_id': i,
                'true_label': sample_label,
                'predicted_label': prediction.is_fraud,
                'fraud_probability': prediction.fraud_probability,
                'asymmetry_score': prediction.asymmetry_score,
                'confidence': prediction.confidence,
                'uncertainty': prediction.uncertainty,
                'processing_time_ms': processing_time
            })
            
            # Print result
            status = "FRAUD" if prediction.is_fraud else "NORMAL"
            true_status = "FRAUD" if sample_label else "NORMAL"
            correct = "✓" if prediction.is_fraud == sample_label else "✗"
            
            print(f"Sample {i:3d}: {status:6s} (True: {true_status:6s}) {correct} | "
                  f"Prob: {prediction.fraud_probability:.3f} | "
                  f"Conf: {prediction.confidence:.3f} | "
                  f"Time: {processing_time:.1f}ms")
            
            # Add delay
            time.sleep(delay_ms / 1000.0)
        
        # Stop inference thread
        self.inference.stop_inference_thread()
        
        # Calculate demo statistics
        self._print_demo_statistics(results)
        
        return results
    
    def run_batch_demo(self, 
                      num_samples: int = 1000):
        """
        Run batch inference demo.
        
        Args:
            num_samples: Number of samples to process
        """
        logger.info("Starting batch inference demo")
        
        # Generate synthetic data
        data = self.generate_synthetic_data(num_samples, fraud_ratio=0.1)
        
        # Create batch inference system
        batch_inference = BatchInference(
            model_path=self.model_path,
            config_path=self.config_path,
            batch_size=32
        )
        
        print("\n" + "="*80)
        print("BATCH FRAUD DETECTION DEMO")
        print("="*80)
        print(f"Processing {num_samples} samples in batches...")
        
        # Run batch inference
        start_time = time.time()
        predictions = batch_inference.predict_batch(
            data['features'],
            data['timestamps'].tolist()
        )
        total_time = time.time() - start_time
        
        # Calculate statistics
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'sample_id': i,
                'true_label': data['labels'][i],
                'predicted_label': pred.is_fraud,
                'fraud_probability': pred.fraud_probability,
                'asymmetry_score': pred.asymmetry_score,
                'confidence': pred.confidence,
                'uncertainty': pred.uncertainty
            })
        
        # Print statistics
        self._print_demo_statistics(results)
        
        print(f"\nBatch Processing:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Throughput: {num_samples/total_time:.1f} samples/second")
        print(f"Average time per sample: {total_time/num_samples*1000:.2f} ms")
        
        return results
    
    def run_explanation_demo(self, 
                            num_samples: int = 10):
        """
        Run model explanation demo.
        
        Args:
            num_samples: Number of samples to explain
        """
        if self.explainer is None:
            logger.warning("Explainer not available, skipping explanation demo")
            return
        
        logger.info("Starting explanation demo")
        
        # Generate synthetic data
        data = self.generate_synthetic_data(num_samples, fraud_ratio=0.3)
        
        print("\n" + "="*80)
        print("MODEL EXPLANATION DEMO")
        print("="*80)
        print(f"Explaining {num_samples} samples...")
        print("-"*80)
        
        for i in range(num_samples):
            sample_features = data['features'][i:i+1]
            sample_label = data['labels'][i]
            
            # Make prediction
            prediction = self.inference.predict_single(sample_features[0])
            
            # Generate explanation
            explanation = self.explainer.explain_shap(sample_features)
            
            # Print explanation
            status = "FRAUD" if prediction.is_fraud else "NORMAL"
            true_status = "FRAUD" if sample_label else "NORMAL"
            
            print(f"\nSample {i+1}: {status} (True: {true_status})")
            print(f"Fraud Probability: {prediction.fraud_probability:.3f}")
            print(f"Asymmetry Score: {prediction.asymmetry_score:.3f}")
            print(f"Confidence: {prediction.confidence:.3f}")
            
            if 'top_features' in explanation:
                print("\nTop 5 Contributing Features:")
                for j, (feature_name, importance) in enumerate(zip(
                    explanation['top_features']['names'][:5],
                    explanation['top_features']['importance'][:5]
                )):
                    print(f"  {j+1}. {feature_name}: {importance:.4f}")
    
    def _print_demo_statistics(self, results: List[Dict[str, Any]]):
        """Print demo statistics."""
        if not results:
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Calculate metrics
        accuracy = (df['predicted_label'] == df['true_label']).mean()
        precision = (df[df['predicted_label'] == True]['true_label'] == True).mean()
        recall = (df[df['true_label'] == True]['predicted_label'] == True).mean()
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate performance metrics
        mean_processing_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 0
        mean_confidence = df['confidence'].mean()
        mean_uncertainty = df['uncertainty'].mean()
        
        print("\n" + "="*80)
        print("DEMO STATISTICS")
        print("="*80)
        print(f"Total Samples: {len(results)}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Mean Processing Time: {mean_processing_time:.2f} ms")
        print(f"Mean Confidence: {mean_confidence:.3f}")
        print(f"Mean Uncertainty: {mean_uncertainty:.3f}")
        
        # Fraud detection breakdown
        fraud_samples = df[df['true_label'] == True]
        if len(fraud_samples) > 0:
            fraud_detection_rate = (fraud_samples['predicted_label'] == True).mean()
            print(f"Fraud Detection Rate: {fraud_detection_rate:.3f}")
        
        # False positive rate
        normal_samples = df[df['true_label'] == False]
        if len(normal_samples) > 0:
            false_positive_rate = (normal_samples['predicted_label'] == True).mean()
            print(f"False Positive Rate: {false_positive_rate:.3f}")
        
        print("="*80)
    
    def create_visualization_demo(self, 
                                 output_dir: str = "demo_visualizations"):
        """Create visualization demo."""
        logger.info("Creating visualization demo")
        
        # Generate synthetic data
        data = self.generate_synthetic_data(1000, fraud_ratio=0.1)
        
        # Run predictions
        batch_inference = BatchInference(
            model_path=self.model_path,
            config_path=self.config_path,
            batch_size=32
        )
        
        predictions = batch_inference.predict_batch(
            data['features'],
            data['timestamps'].tolist()
        )
        
        # Create visualizations
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Fraud Probability Distribution
        plt.figure(figsize=(10, 6))
        fraud_probs = [p.fraud_probability for p in predictions]
        plt.hist(fraud_probs, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.xlabel('Fraud Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Fraud Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'fraud_probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Asymmetry Score vs Fraud Probability
        plt.figure(figsize=(10, 6))
        asymmetry_scores = [p.asymmetry_score for p in predictions]
        colors = ['red' if p.is_fraud else 'blue' for p in predictions]
        
        plt.scatter(asymmetry_scores, fraud_probs, c=colors, alpha=0.6)
        plt.xlabel('Asymmetry Score')
        plt.ylabel('Fraud Probability')
        plt.title('Asymmetry Score vs Fraud Probability')
        plt.colorbar(label='Predicted Label')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'asymmetry_vs_probability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confidence vs Uncertainty
        plt.figure(figsize=(10, 6))
        confidences = [p.confidence for p in predictions]
        uncertainties = [p.uncertainty for p in predictions]
        
        plt.scatter(confidences, uncertainties, c=colors, alpha=0.6)
        plt.xlabel('Confidence')
        plt.ylabel('Uncertainty')
        plt.title('Confidence vs Uncertainty')
        plt.colorbar(label='Predicted Label')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_vs_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Time Series of Predictions
        plt.figure(figsize=(15, 8))
        
        # Plot fraud probabilities over time
        plt.subplot(2, 1, 1)
        plt.plot(data['timestamps'], fraud_probs, alpha=0.7, label='Fraud Probability')
        plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.xlabel('Time')
        plt.ylabel('Fraud Probability')
        plt.title('Fraud Probability Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot asymmetry scores over time
        plt.subplot(2, 1, 2)
        plt.plot(data['timestamps'], asymmetry_scores, alpha=0.7, label='Asymmetry Score')
        plt.xlabel('Time')
        plt.ylabel('Asymmetry Score')
        plt.title('Asymmetry Score Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'time_series_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization demo saved to {output_path}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Dark Pool Fraud Detection Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--demo_type', type=str, 
                       choices=['realtime', 'batch', 'explanation', 'visualization', 'all'],
                       default='all',
                       help='Type of demo to run')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for demo')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                       help='Output directory for demo results')
    
    args = parser.parse_args()
    
    logger.info("Starting Dark Pool Fraud Detection Demo")
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Demo type: {args.demo_type}")
    
    try:
        # Create demo
        demo = FraudDetectionDemo(
            model_path=args.model,
            config_path=args.config
        )
        
        # Run demos based on type
        if args.demo_type in ['realtime', 'all']:
            demo.run_real_time_demo(num_samples=args.num_samples)
        
        if args.demo_type in ['batch', 'all']:
            demo.run_batch_demo(num_samples=args.num_samples * 10)
        
        if args.demo_type in ['explanation', 'all']:
            demo.run_explanation_demo(num_samples=min(args.num_samples, 10))
        
        if args.demo_type in ['visualization', 'all']:
            demo.create_visualization_demo(args.output_dir)
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
