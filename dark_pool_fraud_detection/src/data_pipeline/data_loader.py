"""
Data loading and preprocessing for dark pool trading data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DarkPoolDataset(Dataset):
    """
    Dataset class for dark pool trading data with temporal graph structure.
    """
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 100,
                 features: List[str] = None,
                 is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file
            sequence_length: Length of temporal sequences
            features: List of feature columns to use
            is_training: Whether this is training data
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.features = features
        self.is_training = is_training
        
        # Load and preprocess data
        self.data = self._load_data()
        self.sequences = self._create_sequences()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the raw data."""
        logger.info(f"Loading data from {self.data_path}")
        
        # For now, create synthetic data - replace with actual data loading
        if not Path(self.data_path).exists():
            logger.warning("Data file not found, generating synthetic data")
            return self._generate_synthetic_data()
        
        data = pd.read_parquet(self.data_path)
        
        # Basic preprocessing
        data = self._preprocess_data(data)
        
        return data
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic dark pool trading data for testing."""
        np.random.seed(42)
        n_samples = 100000
        
        # Generate synthetic trading data
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1min'),
            'trader_id': np.random.randint(1, 1000, n_samples),
            'asset_id': np.random.randint(1, 100, n_samples),
            'price': np.random.lognormal(4.5, 0.1, n_samples),
            'volume': np.random.lognormal(8, 1, n_samples),
            'side': np.random.choice(['buy', 'sell'], n_samples),
            'order_type': np.random.choice(['market', 'limit'], n_samples),
            'execution_time': np.random.exponential(0.1, n_samples),
            'latency': np.random.exponential(0.001, n_samples),
            'spread': np.random.exponential(0.01, n_samples),
            'volatility': np.random.gamma(2, 0.1, n_samples),
            'liquidity': np.random.lognormal(10, 1, n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        # Add more sophisticated features
        data['price_change'] = np.diff(data['price'], prepend=data['price'][0])
        data['volume_weighted_price'] = data['price'] * data['volume']
        data['trade_intensity'] = np.random.poisson(5, n_samples)
        data['order_imbalance'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded data."""
        # Handle missing values
        data = data.fillna(method='ffill').fillna(0)
        
        # Convert categorical variables
        if 'side' in data.columns:
            data['side'] = data['side'].map({'buy': 1, 'sell': -1})
        
        if 'order_type' in data.columns:
            data['order_type'] = data['order_type'].map({'market': 1, 'limit': 0})
        
        # Normalize numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['is_fraud', 'trader_id', 'asset_id']:
                data[col] = (data[col] - data[col].mean()) / data[col].std()
        
        return data
    
    def _create_sequences(self) -> List[Dict]:
        """Create temporal sequences from the data."""
        sequences = []
        
        # Group by trader_id and asset_id to create sequences
        grouped = self.data.groupby(['trader_id', 'asset_id'])
        
        for (trader_id, asset_id), group in grouped:
            group = group.sort_values('timestamp')
            
            if len(group) < self.sequence_length:
                continue
                
            # Create sliding windows
            for i in range(len(group) - self.sequence_length + 1):
                sequence = group.iloc[i:i + self.sequence_length]
                
                seq_data = {
                    'features': sequence[self.features].values if self.features else sequence.select_dtypes(include=[np.number]).values,
                    'timestamps': sequence['timestamp'].values,
                    'trader_id': trader_id,
                    'asset_id': asset_id,
                    'label': sequence['is_fraud'].iloc[-1] if 'is_fraud' in sequence.columns else 0,
                    'sequence_id': f"{trader_id}_{asset_id}_{i}"
                }
                
                sequences.append(seq_data)
        
        logger.info(f"Created {len(sequences)} sequences")
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        sequence = self.sequences[idx]
        
        return {
            'features': torch.FloatTensor(sequence['features']),
            'timestamps': torch.LongTensor(sequence['timestamps']),
            'trader_id': torch.LongTensor([sequence['trader_id']]),
            'asset_id': torch.LongTensor([sequence['asset_id']]),
            'label': torch.LongTensor([sequence['label']]),
            'sequence_id': sequence['sequence_id']
        }


class DarkPoolDataLoader:
    """
    Data loader for dark pool trading data with temporal graph structure.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the data loader with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.hardware_config = self.config['hardware']
        
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Define features to use
        features = [
            'price', 'volume', 'side', 'order_type', 'execution_time',
            'latency', 'spread', 'volatility', 'liquidity', 'price_change',
            'volume_weighted_price', 'trade_intensity', 'order_imbalance'
        ]
        
        # Create datasets
        train_dataset = DarkPoolDataset(
            data_path=f"{self.data_config['processed_data_path']}/train.parquet",
            sequence_length=self.data_config['sequence_length'],
            features=features,
            is_training=True
        )
        
        val_dataset = DarkPoolDataset(
            data_path=f"{self.data_config['processed_data_path']}/val.parquet",
            sequence_length=self.data_config['sequence_length'],
            features=features,
            is_training=False
        )
        
        test_dataset = DarkPoolDataset(
            data_path=f"{self.data_config['processed_data_path']}/test.parquet",
            sequence_length=self.data_config['sequence_length'],
            features=features,
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    loader = DarkPoolDataLoader("configs/config.yaml")
    train_loader, val_loader, test_loader = loader.create_dataloaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch features shape: {batch['features'].shape}")
        print(f"Batch labels shape: {batch['label'].shape}")
        break
