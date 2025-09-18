#!/usr/bin/env python3
"""
Setup script for the Dark Pool Fraud Detection System.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_gpu_availability():
    """Check if GPU is available."""
    logger.info("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            logger.warning("⚠ No GPU available, will use CPU")
            return False
    except ImportError:
        logger.warning("⚠ PyTorch not installed yet, cannot check GPU")
        return False


def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    
    # Install PyTorch first (with GPU support if available)
    if check_gpu_availability():
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_command = "pip install torch torchvision torchaudio"
    
    if not run_command(torch_command, "Installing PyTorch"):
        return False
    
    # Install other dependencies
    if not run_command("pip install -r requirements.txt", "Installing other dependencies"):
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    logger.info("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "evaluation_results",
        "demo_results",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    return True


def download_sample_data():
    """Download or generate sample data for testing."""
    logger.info("Setting up sample data...")
    
    # Create a simple script to generate sample data
    sample_data_script = """
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data():
    \"\"\"Generate sample dark pool trading data.\"\"\"
    logger.info("Generating sample data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 10,000 samples
    n_samples = 10000
    sequence_length = 100
    num_features = 118
    
    # Generate timestamps
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    
    # Generate features
    features = np.random.randn(n_samples, sequence_length, num_features)
    
    # Generate labels (5% fraud rate)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Add fraud patterns to fraudulent samples
    fraud_indices = np.where(labels == 1)[0]
    
    for idx in fraud_indices:
        # Add momentum ignition pattern
        if np.random.random() < 0.3:
            spike_start = np.random.randint(0, sequence_length - 10)
            spike_end = spike_start + 10
            features[idx, spike_start:spike_end, 0] += np.random.normal(0, 2, 10)
        
        # Add liquidity fade pattern
        if np.random.random() < 0.4:
            volume_decay = np.linspace(1.0, 0.1, sequence_length)
            features[idx, :, 1] *= volume_decay
        
        # Add unusual order sizes
        if np.random.random() < 0.5:
            large_orders = np.random.choice(sequence_length, size=5, replace=False)
            features[idx, large_orders, 2] *= np.random.uniform(5, 10, 5)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'trader_id': np.random.randint(1, 1000, n_samples),
        'asset_id': np.random.randint(1, 100, n_samples),
        'is_fraud': labels
    })
    
    # Add feature columns
    for i in range(num_features):
        data[f'feature_{i}'] = features[:, :, i].tolist()
    
    # Save data
    data_path = Path('data/processed')
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save splits
    train_data.to_parquet('data/processed/train.parquet', index=False)
    val_data.to_parquet('data/processed/val.parquet', index=False)
    test_data.to_parquet('data/processed/test.parquet', index=False)
    
    logger.info(f"✓ Generated {n_samples} samples")
    logger.info(f"✓ Train samples: {len(train_data)}")
    logger.info(f"✓ Validation samples: {len(val_data)}")
    logger.info(f"✓ Test samples: {len(test_data)}")
    logger.info("✓ Sample data saved to data/processed/")

if __name__ == "__main__":
    generate_sample_data()
"""
    
    # Write and run the script
    with open('generate_sample_data.py', 'w') as f:
        f.write(sample_data_script)
    
    if not run_command("python generate_sample_data.py", "Generating sample data"):
        return False
    
    # Clean up
    os.remove('generate_sample_data.py')
    
    return True


def run_tests():
    """Run basic tests to verify installation."""
    logger.info("Running basic tests...")
    
    test_script = """
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def test_imports():
    \"\"\"Test that all required modules can be imported.\"\"\"
    try:
        from src.models.integrated_model import IntegratedFraudDetectionModel
        from src.data_pipeline.data_loader import DarkPoolDataLoader
        from src.training.trainer import FraudDetectionTrainer
        from src.inference.real_time_inference import RealTimeInference
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    \"\"\"Test that the model can be created.\"\"\"
    try:
        import yaml
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model = IntegratedFraudDetectionModel(config=config)
        print("✓ Model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_data_loading():
    \"\"\"Test that data can be loaded.\"\"\"
    try:
        from src.data_pipeline.data_loader import DarkPoolDataLoader
        loader = DarkPoolDataLoader('configs/config.yaml')
        train_loader, val_loader, test_loader = loader.create_dataloaders()
        print("✓ Data loaders created successfully")
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

if __name__ == "__main__":
    print("Running installation tests...")
    print("-" * 40)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_loading
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{len(tests)}")
    if passed == len(tests):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
"""
    
    # Write and run the test script
    with open('test_installation.py', 'w') as f:
        f.write(test_script)
    
    success = run_command("python test_installation.py", "Running installation tests")
    
    # Clean up
    os.remove('test_installation.py')
    
    return success


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Dark Pool Fraud Detection System')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip sample data generation')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip installation tests')
    
    args = parser.parse_args()
    
    logger.info("Setting up Dark Pool Fraud Detection System")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("Dependency installation failed")
            sys.exit(1)
    
    # Generate sample data
    if not args.skip_data:
        if not download_sample_data():
            logger.error("Sample data generation failed")
            sys.exit(1)
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            logger.error("Installation tests failed")
            sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("✓ Setup completed successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Train the model: python train.py")
    logger.info("2. Run evaluation: python evaluate.py --model models/best_model.pth")
    logger.info("3. Run demo: python demo.py --model models/best_model.pth")
    logger.info("")
    logger.info("For more information, see README.md")


if __name__ == "__main__":
    main()
