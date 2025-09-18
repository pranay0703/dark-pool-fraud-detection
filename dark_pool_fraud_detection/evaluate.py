#!/usr/bin/env python3
"""
Evaluation script for the dark pool fraud detection system.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Any
import json
import time

from src.training.trainer import FraudDetectionTrainer
from src.data_pipeline.data_loader import DarkPoolDataLoader
from src.inference.real_time_inference import RealTimeInference, BatchInference

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for fraud detection.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 test_data_path: Optional[str] = None):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            test_data_path: Path to test data (optional)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.test_data_path = test_data_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        
        # Load test data
        self.test_loader = self._load_test_data()
        
        logger.info("Model evaluator initialized")
    
    def _load_model(self):
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
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def _load_test_data(self):
        """Load test data."""
        if self.test_data_path:
            # Load custom test data
            data_loader = DarkPoolDataLoader(self.config_path)
            _, _, test_loader = data_loader.create_dataloaders()
            return test_loader
        else:
            # Use default test data
            data_loader = DarkPoolDataLoader(self.config_path)
            _, _, test_loader = data_loader.create_dataloaders()
            return test_loader
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting model evaluation...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_asymmetry_scores = []
        all_confidences = []
        all_uncertainties = []
        
        processing_times = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                start_time = time.time()
                
                # Move batch to device
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                processing_times.append(processing_time)
                
                # Store results
                predictions = (torch.sigmoid(outputs['fraud_logits'].squeeze()) > 0.5).cpu().numpy()
                probabilities = torch.sigmoid(outputs['fraud_probability']).cpu().numpy()
                asymmetry_scores = outputs['asymmetry_score'].cpu().numpy()
                confidences = outputs['confidence'].cpu().numpy()
                uncertainties = outputs['uncertainty'].cpu().numpy() if 'uncertainty' in outputs else np.zeros_like(probabilities)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities)
                all_asymmetry_scores.extend(asymmetry_scores)
                all_confidences.extend(confidences)
                all_uncertainties.extend(uncertainties)
                
                if batch_idx % 100 == 0:
                    logger.info(f"Processed {batch_idx} batches")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        all_asymmetry_scores = np.array(all_asymmetry_scores)
        all_confidences = np.array(all_confidences)
        all_uncertainties = np.array(all_uncertainties)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, all_predictions, all_probabilities,
            all_asymmetry_scores, all_confidences, all_uncertainties
        )
        
        # Add performance metrics
        metrics['performance'] = {
            'mean_processing_time_ms': np.mean(processing_times),
            'median_processing_time_ms': np.median(processing_times),
            'p95_processing_time_ms': np.percentile(processing_times, 95),
            'p99_processing_time_ms': np.percentile(processing_times, 99),
            'max_processing_time_ms': np.max(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'std_processing_time_ms': np.std(processing_times),
            'total_samples': len(all_labels)
        }
        
        logger.info("Model evaluation completed")
        return metrics
    
    def _calculate_metrics(self, 
                          labels: np.ndarray,
                          predictions: np.ndarray,
                          probabilities: np.ndarray,
                          asymmetry_scores: np.ndarray,
                          confidences: np.ndarray,
                          uncertainties: np.ndarray) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        
        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(labels, probabilities)
        except ValueError:
            auc_roc = 0.0
        
        # Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, probabilities)
        auc_pr = np.trapz(precision_curve, recall_curve)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(labels, probabilities)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        class_report = classification_report(labels, predictions, output_dict=True)
        
        # Asymmetry score analysis
        fraud_asymmetry = asymmetry_scores[labels == 1]
        normal_asymmetry = asymmetry_scores[labels == 0]
        
        # Confidence analysis
        fraud_confidence = confidences[labels == 1]
        normal_confidence = confidences[labels == 0]
        
        # Uncertainty analysis
        fraud_uncertainty = uncertainties[labels == 1]
        normal_uncertainty = uncertainties[labels == 0]
        
        return {
            'classification': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'curves': {
                'precision_curve': precision_curve.tolist(),
                'recall_curve': recall_curve.tolist(),
                'pr_thresholds': pr_thresholds.tolist(),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'roc_thresholds': roc_thresholds.tolist()
            },
            'asymmetry_analysis': {
                'fraud_mean': np.mean(fraud_asymmetry) if len(fraud_asymmetry) > 0 else 0.0,
                'fraud_std': np.std(fraud_asymmetry) if len(fraud_asymmetry) > 0 else 0.0,
                'normal_mean': np.mean(normal_asymmetry) if len(normal_asymmetry) > 0 else 0.0,
                'normal_std': np.std(normal_asymmetry) if len(normal_asymmetry) > 0 else 0.0
            },
            'confidence_analysis': {
                'fraud_mean': np.mean(fraud_confidence) if len(fraud_confidence) > 0 else 0.0,
                'fraud_std': np.std(fraud_confidence) if len(fraud_confidence) > 0 else 0.0,
                'normal_mean': np.mean(normal_confidence) if len(normal_confidence) > 0 else 0.0,
                'normal_std': np.std(normal_confidence) if len(normal_confidence) > 0 else 0.0
            },
            'uncertainty_analysis': {
                'fraud_mean': np.mean(fraud_uncertainty) if len(fraud_uncertainty) > 0 else 0.0,
                'fraud_std': np.std(fraud_uncertainty) if len(fraud_uncertainty) > 0 else 0.0,
                'normal_mean': np.mean(normal_uncertainty) if len(normal_uncertainty) > 0 else 0.0,
                'normal_std': np.std(normal_uncertainty) if len(normal_uncertainty) > 0 else 0.0
            }
        }
    
    def plot_evaluation_results(self, 
                               metrics: Dict[str, Any],
                               output_dir: str):
        """Plot evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr = metrics['curves']['fpr']
        tpr = metrics['curves']['tpr']
        auc_roc = metrics['classification']['auc_roc']
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision_curve = metrics['curves']['precision_curve']
        recall_curve = metrics['curves']['recall_curve']
        auc_pr = metrics['classification']['auc_pr']
        
        plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Asymmetry Score Distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(metrics['asymmetry_analysis']['normal_mean'], 
                bins=50, alpha=0.7, label='Normal', color='blue')
        plt.hist(metrics['asymmetry_analysis']['fraud_mean'], 
                bins=50, alpha=0.7, label='Fraud', color='red')
        plt.xlabel('Asymmetry Score')
        plt.ylabel('Frequency')
        plt.title('Asymmetry Score Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(metrics['confidence_analysis']['normal_mean'], 
                bins=50, alpha=0.7, label='Normal', color='blue')
        plt.hist(metrics['confidence_analysis']['fraud_mean'], 
                bins=50, alpha=0.7, label='Fraud', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {output_path}")
    
    def save_results(self, 
                    metrics: Dict[str, Any],
                    output_dir: str):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        with open(output_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save classification report as text
        with open(output_path / 'classification_report.txt', 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Accuracy: {metrics['classification']['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['classification']['precision']:.4f}\n")
            f.write(f"Recall: {metrics['classification']['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['classification']['f1_score']:.4f}\n")
            f.write(f"AUC-ROC: {metrics['classification']['auc_roc']:.4f}\n")
            f.write(f"AUC-PR: {metrics['classification']['auc_pr']:.4f}\n\n")
            
            f.write("Performance Metrics\n")
            f.write("=" * 50 + "\n")
            perf = metrics['performance']
            f.write(f"Mean Processing Time: {perf['mean_processing_time_ms']:.2f} ms\n")
            f.write(f"Median Processing Time: {perf['median_processing_time_ms']:.2f} ms\n")
            f.write(f"95th Percentile: {perf['p95_processing_time_ms']:.2f} ms\n")
            f.write(f"99th Percentile: {perf['p99_processing_time_ms']:.2f} ms\n")
            f.write(f"Total Samples: {perf['total_samples']}\n")
        
        logger.info(f"Evaluation results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Dark Pool Fraud Detection Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data (optional)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    logger.info("Starting model evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(
            model_path=args.model,
            config_path=args.config,
            test_data_path=args.test_data
        )
        
        # Run evaluation
        metrics = evaluator.evaluate_model()
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"Accuracy: {metrics['classification']['accuracy']:.4f}")
        print(f"Precision: {metrics['classification']['precision']:.4f}")
        print(f"Recall: {metrics['classification']['recall']:.4f}")
        print(f"F1 Score: {metrics['classification']['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['classification']['auc_roc']:.4f}")
        print(f"AUC-PR: {metrics['classification']['auc_pr']:.4f}")
        
        print(f"\nPerformance:")
        perf = metrics['performance']
        print(f"Mean Processing Time: {perf['mean_processing_time_ms']:.2f} ms")
        print(f"95th Percentile: {perf['p95_processing_time_ms']:.2f} ms")
        print(f"Total Samples: {perf['total_samples']}")
        
        print("="*60)
        
        # Save results
        evaluator.save_results(metrics, args.output_dir)
        
        # Generate plots if requested
        if args.plot:
            evaluator.plot_evaluation_results(metrics, args.output_dir)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
