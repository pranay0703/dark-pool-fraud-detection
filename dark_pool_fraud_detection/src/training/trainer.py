"""
Training pipeline for the integrated fraud detection model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.integrated_model import IntegratedFraudDetectionModel
from ..data_pipeline.data_loader import DarkPoolDataLoader
from ..data_pipeline.temporal_graph import TemporalGraphBuilder

logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """
    Trainer class for the integrated fraud detection model.
    """
    
    def __init__(self, 
                 config_path: str,
                 model: Optional[IntegratedFraudDetectionModel] = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
            model: Pre-trained model (if None, will create new model)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.model_config = self.config['model']
        self.evaluation_config = self.config['evaluation']
        self.hardware_config = self.config['hardware']
        self.logging_config = self.config['logging']
        
        # Set device
        self.device = torch.device(self.hardware_config['device'] if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        if model is None:
            self.model = IntegratedFraudDetectionModel(
                config=self.config,
                feature_names=[f"feature_{i}" for i in range(self.config['data']['num_features'])]
            )
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss functions
        self.loss_functions = self._create_loss_functions()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Create output directories
        self._create_directories()
        
        # Initialize logging
        self._setup_logging()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.training_config['optimizer'].lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.training_config['scheduler'].lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config['epochs'] // 3,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _create_loss_functions(self) -> Dict[str, nn.Module]:
        """Create loss functions."""
        loss_weights = self.training_config['loss_weights']
        
        return {
            'classification': nn.BCEWithLogitsLoss(),
            'reconstruction': nn.MSELoss(),
            'temporal_consistency': nn.MSELoss()
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        self.model_dir = Path(self.logging_config['log_dir']) / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.logging_config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.logging_config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_reconstruction_loss = 0.0
        total_temporal_loss = 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device).float()
            
            # Create temporal graph if needed
            edge_index = None
            batch_assignment = None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(features, edge_index, batch_assignment)
            
            # Calculate losses
            classification_loss = self.loss_functions['classification'](
                outputs['fraud_logits'].squeeze(), labels
            )
            
            # Reconstruction loss (if applicable)
            reconstruction_loss = 0.0
            if 'reconstruction' in outputs:
                reconstruction_loss = self.loss_functions['reconstruction'](
                    outputs['reconstruction'], features
                )
            
            # Temporal consistency loss
            temporal_loss = 0.0
            if 'temporal_features' in outputs:
                # Simple temporal consistency: consecutive features should be similar
                if features.size(1) > 1:
                    temporal_diff = features[:, 1:] - features[:, :-1]
                    temporal_consistency = outputs['temporal_features'].unsqueeze(1).repeat(1, features.size(1)-1, 1)
                    temporal_loss = self.loss_functions['temporal_consistency'](
                        temporal_consistency, temporal_diff
                    )
            
            # Total loss
            total_loss_batch = (
                self.training_config['loss_weights']['classification'] * classification_loss +
                self.training_config['loss_weights']['reconstruction'] * reconstruction_loss +
                self.training_config['loss_weights']['temporal_consistency'] * temporal_loss
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.training_config.get('gradient_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_classification_loss += classification_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_temporal_loss += temporal_loss.item()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs['fraud_logits'].squeeze()) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Store predictions for metrics
            all_predictions.extend(torch.sigmoid(outputs['fraud_logits'].squeeze()).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss_batch.item():.4f}",
                'Acc': f"{correct_predictions/total_predictions:.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_classification_loss = total_classification_loss / len(train_loader)
        epoch_reconstruction_loss = total_reconstruction_loss / len(train_loader)
        epoch_temporal_loss = total_temporal_loss / len(train_loader)
        
        epoch_accuracy = correct_predictions / total_predictions
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # F1 score
        predictions_binary = (all_predictions > 0.5).astype(int)
        f1_score = self._calculate_f1_score(all_labels, predictions_binary)
        
        # AUC-ROC
        auc_roc = self._calculate_auc_roc(all_labels, all_predictions)
        
        return {
            'loss': epoch_loss,
            'classification_loss': epoch_classification_loss,
            'reconstruction_loss': epoch_reconstruction_loss,
            'temporal_loss': epoch_temporal_loss,
            'accuracy': epoch_accuracy,
            'f1_score': f1_score,
            'auc_roc': auc_roc
        }
    
    def validate_epoch(self, 
                      val_loader: DataLoader,
                      epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_reconstruction_loss = 0.0
        total_temporal_loss = 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device).float()
                
                # Create temporal graph if needed
                edge_index = None
                batch_assignment = None
                
                # Forward pass
                outputs = self.model(features, edge_index, batch_assignment)
                
                # Calculate losses
                classification_loss = self.loss_functions['classification'](
                    outputs['fraud_logits'].squeeze(), labels
                )
                
                # Reconstruction loss (if applicable)
                reconstruction_loss = 0.0
                if 'reconstruction' in outputs:
                    reconstruction_loss = self.loss_functions['reconstruction'](
                        outputs['reconstruction'], features
                    )
                
                # Temporal consistency loss
                temporal_loss = 0.0
                if 'temporal_features' in outputs:
                    if features.size(1) > 1:
                        temporal_diff = features[:, 1:] - features[:, :-1]
                        temporal_consistency = outputs['temporal_features'].unsqueeze(1).repeat(1, features.size(1)-1, 1)
                        temporal_loss = self.loss_functions['temporal_consistency'](
                            temporal_consistency, temporal_diff
                        )
                
                # Total loss
                total_loss_batch = (
                    self.training_config['loss_weights']['classification'] * classification_loss +
                    self.training_config['loss_weights']['reconstruction'] * reconstruction_loss +
                    self.training_config['loss_weights']['temporal_consistency'] * temporal_loss
                )
                
                # Update metrics
                total_loss += total_loss_batch.item()
                total_classification_loss += classification_loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_temporal_loss += temporal_loss.item()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs['fraud_logits'].squeeze()) > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Store predictions for metrics
                all_predictions.extend(torch.sigmoid(outputs['fraud_logits'].squeeze()).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(val_loader)
        epoch_classification_loss = total_classification_loss / len(val_loader)
        epoch_reconstruction_loss = total_reconstruction_loss / len(val_loader)
        epoch_temporal_loss = total_temporal_loss / len(val_loader)
        
        epoch_accuracy = correct_predictions / total_predictions
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # F1 score
        predictions_binary = (all_predictions > 0.5).astype(int)
        f1_score = self._calculate_f1_score(all_labels, predictions_binary)
        
        # AUC-ROC
        auc_roc = self._calculate_auc_roc(all_labels, all_predictions)
        
        return {
            'loss': epoch_loss,
            'classification_loss': epoch_classification_loss,
            'reconstruction_loss': epoch_reconstruction_loss,
            'temporal_loss': epoch_temporal_loss,
            'accuracy': epoch_accuracy,
            'f1_score': f1_score,
            'auc_roc': auc_roc
        }
    
    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='binary')
    
    def _calculate_auc_roc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUC-ROC score."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.training_config['epochs']):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1_score'])
            self.training_history['val_f1'].append(val_metrics['f1_score'])
            self.training_history['train_auc'].append(train_metrics['auc_roc'])
            self.training_history['val_auc'].append(val_metrics['auc_roc'])
            self.training_history['learning_rates'].append(current_lr)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.training_config['epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Train F1: {train_metrics['f1_score']:.4f}, "
                f"Val F1: {val_metrics['f1_score']:.4f}, "
                f"Train AUC: {train_metrics['auc_roc']:.4f}, "
                f"Val AUC: {val_metrics['auc_roc']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                
                # Save best model
                self.save_model(self.model_dir / 'best_model.pth')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.training_config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.logging_config.get('checkpoint_frequency', 10) == 0:
                self.save_model(self.model_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Training completed
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        if test_loader is not None:
            test_metrics = self.validate_epoch(test_loader, epoch)
            logger.info(f"Test metrics: {test_metrics}")
        
        # Save final model
        self.save_model(self.model_dir / 'final_model.pth')
        
        # Save training history
        self.save_training_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        return {
            'training_time': training_time,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'final_epoch': epoch + 1,
            'training_history': self.training_history
        }
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        logger.info(f"Model loaded from {path}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score curves
        axes[0, 2].plot(self.training_history['train_f1'], label='Train F1')
        axes[0, 2].plot(self.training_history['val_f1'], label='Val F1')
        axes[0, 2].set_title('F1 Score Curves')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # AUC curves
        axes[1, 0].plot(self.training_history['train_auc'], label='Train AUC')
        axes[1, 0].plot(self.training_history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('AUC Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        axes[1, 1].plot(self.training_history['learning_rates'])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training curves saved")


if __name__ == "__main__":
    # Test the trainer
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = FraudDetectionTrainer('configs/config.yaml')
    
    # Create sample data loaders
    data_loader = DarkPoolDataLoader('configs/config.yaml')
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    
    # Train model
    results = trainer.train(train_loader, val_loader, test_loader)
    
    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Training time: {results['training_time']:.2f} seconds")
