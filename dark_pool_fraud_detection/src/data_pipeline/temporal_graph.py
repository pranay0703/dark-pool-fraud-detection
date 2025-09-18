"""
Event-Based Temporal Graph (ETG) construction for dark pool trading data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, from_networkx
import logging

logger = logging.getLogger(__name__)


class EventBasedTemporalGraph:
    """
    Constructs and manages Event-Based Temporal Graphs for dark pool trading data.
    """
    
    def __init__(self, 
                 node_types: List[str] = None,
                 edge_types: List[str] = None,
                 time_window: int = 3600):  # 1 hour in seconds
        """
        Initialize the ETG constructor.
        
        Args:
            node_types: Types of nodes in the graph (traders, assets, accounts, etc.)
            edge_types: Types of edges (trades, communications, etc.)
            time_window: Time window for temporal aggregation (seconds)
        """
        self.node_types = node_types or ['trader', 'asset', 'account', 'device', 'ip']
        self.edge_types = edge_types or ['trade', 'communication', 'login', 'registration']
        self.time_window = time_window
        
        # Node and edge mappings
        self.node_mappings = {node_type: {} for node_type in self.node_types}
        self.edge_mappings = {edge_type: {} for edge_type in self.edge_types}
        
        # Global node and edge counters
        self.global_node_id = 0
        self.global_edge_id = 0
        
    def add_trading_events(self, 
                          trading_data: pd.DataFrame,
                          timestamp_col: str = 'timestamp') -> Dict:
        """
        Add trading events to the temporal graph.
        
        Args:
            trading_data: DataFrame containing trading events
            timestamp_col: Name of the timestamp column
            
        Returns:
            Dictionary containing graph construction metadata
        """
        logger.info(f"Adding {len(trading_data)} trading events to ETG")
        
        events = []
        current_time = None
        time_bucket = 0
        
        for _, row in trading_data.iterrows():
            timestamp = row[timestamp_col]
            
            # Create time buckets
            if current_time is None or (timestamp - current_time).total_seconds() > self.time_window:
                time_bucket += 1
                current_time = timestamp
            
            # Create trade event
            event = self._create_trade_event(row, time_bucket)
            events.append(event)
        
        # Build temporal graph from events
        graph_data = self._build_temporal_graph(events)
        
        return {
            'num_events': len(events),
            'num_time_buckets': time_bucket,
            'graph_data': graph_data
        }
    
    def _create_trade_event(self, row: pd.Series, time_bucket: int) -> Dict:
        """Create a trade event from a data row."""
        # Get or create node IDs
        trader_id = self._get_or_create_node('trader', row['trader_id'])
        asset_id = self._get_or_create_node('asset', row['asset_id'])
        
        # Create trade edge
        trade_edge = {
            'edge_type': 'trade',
            'source': trader_id,
            'target': asset_id,
            'timestamp': row['timestamp'],
            'time_bucket': time_bucket,
            'features': {
                'price': row['price'],
                'volume': row['volume'],
                'side': row['side'],
                'order_type': row['order_type'],
                'execution_time': row['execution_time'],
                'latency': row['latency'],
                'spread': row['spread'],
                'volatility': row['volatility'],
                'liquidity': row['liquidity']
            }
        }
        
        return trade_edge
    
    def _get_or_create_node(self, node_type: str, node_value: Union[str, int]) -> int:
        """Get existing node ID or create new one."""
        if node_value not in self.node_mappings[node_type]:
            self.node_mappings[node_type][node_value] = self.global_node_id
            self.global_node_id += 1
        
        return self.node_mappings[node_type][node_value]
    
    def _build_temporal_graph(self, events: List[Dict]) -> Data:
        """Build PyTorch Geometric Data object from events."""
        # Collect all nodes and edges
        nodes = set()
        edges = []
        edge_attrs = []
        node_attrs = []
        
        # Process events to build graph structure
        for event in events:
            source = event['source']
            target = event['target']
            
            nodes.add(source)
            nodes.add(target)
            
            edges.append([source, target])
            
            # Edge attributes
            edge_attr = list(event['features'].values())
            edge_attrs.append(edge_attr)
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Create node attributes (simplified for now)
        num_nodes = len(nodes)
        node_attr = torch.randn(num_nodes, 64)  # Random node features
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        return data
    
    def get_temporal_subgraph(self, 
                             start_time: pd.Timestamp,
                             end_time: pd.Timestamp) -> Data:
        """
        Extract a temporal subgraph for a specific time period.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            PyTorch Geometric Data object for the subgraph
        """
        # This would filter events by time and rebuild the graph
        # For now, return the full graph
        logger.info(f"Extracting subgraph from {start_time} to {end_time}")
        return self.graph_data
    
    def get_node_embeddings(self, 
                           node_type: str,
                           time_bucket: int) -> torch.Tensor:
        """
        Get node embeddings for a specific node type and time bucket.
        
        Args:
            node_type: Type of nodes to get embeddings for
            time_bucket: Time bucket index
            
        Returns:
            Tensor of node embeddings
        """
        # This would return learned node embeddings
        # For now, return random embeddings
        num_nodes = len(self.node_mappings[node_type])
        return torch.randn(num_nodes, 128)


class TemporalGraphBuilder:
    """
    Builder class for constructing complex temporal graphs from trading data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the temporal graph builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.etg = EventBasedTemporalGraph(
            node_types=config.get('node_types', ['trader', 'asset', 'account']),
            edge_types=config.get('edge_types', ['trade', 'communication']),
            time_window=config.get('time_window', 3600)
        )
    
    def build_from_trading_data(self, 
                               trading_data: pd.DataFrame) -> Data:
        """
        Build temporal graph from trading data.
        
        Args:
            trading_data: DataFrame containing trading events
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Building temporal graph from trading data")
        
        # Add trading events
        result = self.etg.add_trading_events(trading_data)
        
        # Store the graph data
        self.etg.graph_data = result['graph_data']
        
        return result['graph_data']
    
    def add_communication_events(self, 
                                comm_data: pd.DataFrame) -> None:
        """
        Add communication events to the existing graph.
        
        Args:
            comm_data: DataFrame containing communication events
        """
        logger.info("Adding communication events to temporal graph")
        
        # This would add communication edges between traders
        # For now, this is a placeholder
        pass
    
    def add_device_events(self, 
                         device_data: pd.DataFrame) -> None:
        """
        Add device and IP address events to the graph.
        
        Args:
            device_data: DataFrame containing device events
        """
        logger.info("Adding device events to temporal graph")
        
        # This would add device and IP nodes and edges
        # For now, this is a placeholder
        pass
    
    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the constructed graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        if self.etg.graph_data is None:
            return {}
        
        return {
            'num_nodes': self.etg.graph_data.num_nodes,
            'num_edges': self.etg.graph_data.edge_index.size(1),
            'num_node_types': len(self.etg.node_types),
            'num_edge_types': len(self.etg.edge_types),
            'node_degree_stats': self._calculate_degree_stats()
        }
    
    def _calculate_degree_stats(self) -> Dict:
        """Calculate degree statistics for the graph."""
        if self.etg.graph_data is None:
            return {}
        
        edge_index = self.etg.graph_data.edge_index
        degrees = torch.zeros(self.etg.graph_data.num_nodes)
        
        for i in range(edge_index.size(1)):
            source, target = edge_index[:, i]
            degrees[source] += 1
            degrees[target] += 1
        
        return {
            'mean_degree': degrees.mean().item(),
            'std_degree': degrees.std().item(),
            'max_degree': degrees.max().item(),
            'min_degree': degrees.min().item()
        }


if __name__ == "__main__":
    # Test the temporal graph construction
    import pandas as pd
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
        'trader_id': np.random.randint(1, 100, 1000),
        'asset_id': np.random.randint(1, 50, 1000),
        'price': np.random.lognormal(4.5, 0.1, 1000),
        'volume': np.random.lognormal(8, 1, 1000),
        'side': np.random.choice([1, -1], 1000),
        'order_type': np.random.choice([0, 1], 1000),
        'execution_time': np.random.exponential(0.1, 1000),
        'latency': np.random.exponential(0.001, 1000),
        'spread': np.random.exponential(0.01, 1000),
        'volatility': np.random.gamma(2, 0.1, 1000),
        'liquidity': np.random.lognormal(10, 1, 1000)
    })
    
    # Build temporal graph
    config = {
        'node_types': ['trader', 'asset'],
        'edge_types': ['trade'],
        'time_window': 3600
    }
    
    builder = TemporalGraphBuilder(config)
    graph_data = builder.build_from_trading_data(data)
    
    print(f"Graph nodes: {graph_data.num_nodes}")
    print(f"Graph edges: {graph_data.edge_index.size(1)}")
    print(f"Node features shape: {graph_data.x.shape}")
    print(f"Edge features shape: {graph_data.edge_attr.shape}")
    
    stats = builder.get_graph_statistics()
    print(f"Graph statistics: {stats}")
