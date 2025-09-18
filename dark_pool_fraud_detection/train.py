#!/usr/bin/env python3
"""
Main training script for the dark pool fraud detection system.
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np
import random

from src.training.trainer import FraudDetectionTrainer
from src.data_pipeline.data_loader import DarkPoolDataLoader
from src.data_pipeline.temporal_graph import TemporalGraphBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Dark Pool Fraud Detection Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for logs and models (overrides config)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_path:
        config['data']['raw_data_path'] = args.data_path
        config['data']['processed_data_path'] = args.data_path
    
    if args.output_dir:
        config['logging']['log_dir'] = args.output_dir
    
    # Create output directories
    output_dir = Path(config['logging']['log_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Starting Dark Pool Fraud Detection Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loader = DarkPoolDataLoader(args.config)
        train_loader, val_loader, test_loader = data_loader.create_dataloaders()
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = FraudDetectionTrainer(args.config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_model(args.resume)
        
        # Print model summary
        model_summary = trainer.model.get_model_summary()
        logger.info("Model Summary:")
        for key, value in model_summary.items():
            logger.info(f"  {key}: {value:,}")
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train(train_loader, val_loader, test_loader)
        
        # Print final results
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")
        logger.info(f"Final epoch: {results['final_epoch']}")
        
        # Save final results
        import json
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / 'training_results.json'}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
