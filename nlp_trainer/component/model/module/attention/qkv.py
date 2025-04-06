import torch
import torch.nn as nn
from typing import Optional

class QKVCreator(nn.Module):
    
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        positional_encoder: nn.Module,
    ):
        super(QKVCreator, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.positional_encoder = positional_encoder
        
    def forward(
        self,
        x: torch.Tensor,
        cache_key: Optional[torch.Tensor] = None,
        cache_value: Optional[torch.Tensor] = None
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(x.shape) == 3
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM = x.shape

        ### 1. linear projection
        ###
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, self.head_dim)
        k = k.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, self.head_dim)
        v = v.reshape(BATCH_SIZE, SEQ_LEN, self.num_heads, self.head_dim)

        ### 2. positional encoding
        q = self.positional_encoder(q).transpose(1, 2)

        if cache_key is not None and cache_value is not None:
            cache_k, cache_v = cache
            k = self.positional_encoder(k, offset=cache_k.shape[1])

            k = torch.cat((cache_k, k), dim=1)
            v = torch.cat((cache_v, v), dim=1)

            cache = (k, v)

        k, v = k.transpose(1, 2), v.transpose(1, 2)
        
        assert q.shape == (BATCH_SIZE, self.num_heads, SEQ_LEN, self.head_dim)
        
        return q, k, v