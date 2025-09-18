"""
Temporal Graph Neural Network (TGNN) for dark pool fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryModule(nn.Module):
    """
    Memory module for storing and updating node states over time.
    """
    
    def __init__(self, 
                 node_dim: int,
                 memory_dim: int,
                 update_dim: int):
        """
        Initialize the memory module.
        
        Args:
            node_dim: Dimension of node features
            memory_dim: Dimension of memory vectors
            update_dim: Dimension of update vectors
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.memory_dim = memory_dim
        self.update_dim = update_dim
        
        # Memory update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + memory_dim + update_dim, memory_dim * 2),
            nn.ReLU(),
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid()
        )
        
        # Memory reset network
        self.reset_net = nn.Sequential(
            nn.Linear(node_dim + memory_dim + update_dim, memory_dim),
            nn.Sigmoid()
        )
        
        # New memory network
        self.new_net = nn.Sequential(
            nn.Linear(node_dim + memory_dim + update_dim, memory_dim),
            nn.Tanh()
        )
    
    def forward(self, 
                node_features: torch.Tensor,
                memory: torch.Tensor,
                update_vector: torch.Tensor) -> torch.Tensor:
        """
        Update memory based on node features and update vector.
        
        Args:
            node_features: Current node features [num_nodes, node_dim]
            memory: Current memory state [num_nodes, memory_dim]
            update_vector: Update vector from neighbors [num_nodes, update_dim]
            
        Returns:
            Updated memory state [num_nodes, memory_dim]
        """
        # Concatenate inputs
        combined = torch.cat([node_features, memory, update_vector], dim=-1)
        
        # Calculate update, reset, and new memory
        update_gate = self.update_net(combined)
        reset_gate = self.reset_net(combined)
        new_memory = self.new_net(combined)
        
        # Update memory
        updated_memory = update_gate * memory + (1 - update_gate) * new_memory
        
        return updated_memory


class TemporalGraphLayer(nn.Module):
    """
    Single layer of the Temporal Graph Neural Network.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_memory: bool = True):
        """
        Initialize the temporal graph layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_memory: Whether to use memory module
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_memory = use_memory
        
        # Graph attention layer
        self.gat = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # Memory module
        if use_memory:
            self.memory = MemoryModule(
                node_dim=input_dim,
                memory_dim=hidden_dim,
                update_dim=hidden_dim
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the temporal graph layer.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            memory: Previous memory state [num_nodes, memory_dim]
            
        Returns:
            Tuple of (updated_features, updated_memory)
        """
        # Graph attention
        h = self.gat(x, edge_index)
        h = self.dropout(h)
        
        # Residual connection
        if self.residual_proj is not None:
            h = h + self.residual_proj(x)
        else:
            h = h + x
        
        # Layer normalization
        h = self.layer_norm(h)
        
        # Update memory if enabled
        if self.use_memory and memory is not None:
            updated_memory = self.memory(x, memory, h)
        else:
            updated_memory = memory
        
        return h, updated_memory


class TemporalGraphNeuralNetwork(nn.Module):
    """
    Main Temporal Graph Neural Network for fraud detection.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_memory: bool = True,
                 memory_size: int = 1000):
        """
        Initialize the TGNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of graph layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_memory: Whether to use memory module
            memory_size: Size of memory buffer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_memory = use_memory
        self.memory_size = memory_size
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal graph layers
        self.layers = nn.ModuleList([
            TemporalGraphLayer(
                input_dim=hidden_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_memory=use_memory
            )
            for i in range(num_layers)
        ])
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # mean + max + sum
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize memory buffer
        if use_memory:
            self.register_buffer('memory_buffer', torch.zeros(memory_size, hidden_dim))
            self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the TGNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Fraud prediction logits [batch_size, 1]
        """
        # Input projection
        h = self.input_proj(x)
        
        # Initialize memory
        memory = None
        if self.use_memory:
            memory = self._get_memory(x.size(0))
        
        # Pass through temporal graph layers
        for layer in self.layers:
            h, memory = layer(h, edge_index, memory)
        
        # Update memory buffer
        if self.use_memory and memory is not None:
            self._update_memory(memory)
        
        # Global pooling
        if batch is not None:
            # Batch-wise pooling
            mean_pool = global_mean_pool(h, batch)
            max_pool = global_max_pool(h, batch)
            sum_pool = global_add_pool(h, batch)
        else:
            # Global pooling for single graph
            mean_pool = h.mean(dim=0, keepdim=True)
            max_pool = h.max(dim=0, keepdim=True)[0]
            sum_pool = h.sum(dim=0, keepdim=True)
        
        # Concatenate pooling results
        pooled = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        pooled = self.global_pool(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def _get_memory(self, num_nodes: int) -> torch.Tensor:
        """Get memory for current nodes."""
        if not self.use_memory:
            return None
        
        # For simplicity, return zeros - in practice, you'd retrieve from buffer
        return torch.zeros(num_nodes, self.hidden_dim, device=x.device)
    
    def _update_memory(self, memory: torch.Tensor) -> None:
        """Update memory buffer."""
        if not self.use_memory:
            return
        
        # For simplicity, this is a placeholder
        # In practice, you'd update the memory buffer with new states
        pass


class TemporalGraphAttention(nn.Module):
    """
    Multi-head attention mechanism for temporal graph data.
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize temporal graph attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal attention.
        
        Args:
            x: Input features [seq_len, num_nodes, hidden_dim]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Attended features [seq_len, num_nodes, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        # Output projection
        output = self.out_linear(attended)
        
        return output


if __name__ == "__main__":
    # Test the TGNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample data
    num_nodes = 100
    num_edges = 200
    input_dim = 64
    hidden_dim = 128
    
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Create model
    model = TemporalGraphNeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        use_memory=True
    )
    
    # Forward pass
    logits = model(x, edge_index, batch)
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
