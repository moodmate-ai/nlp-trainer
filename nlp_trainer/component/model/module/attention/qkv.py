import torch
import torch.nn as nn

class MultiHeadQKVCreator(nn.Module):
    
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        positional_encoder: nn.Module,
    ):
        super(MultiHeadQKVCreator, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.positional_encoder = positional_encoder
        
    def forward(
        self,
        x: torch.Tensor,
        cache_key: torch.Tensor,
        cache_value: torch.Tensor
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        
        cache_key: (BATCH_SIZE, CACHE_LEN, NUM_HEADS, HEAD_DIM)
        cache_value: (BATCH_SIZE, CACHE_LEN, NUM_HEADS, HEAD_DIM)
        
        returns:
            q: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
            k: (BATCH_SIZE, NUM_HEADS, SEQ_LEN + cache_length, HEAD_DIM)
            v: (BATCH_SIZE, NUM_HEADS, SEQ_LEN + cache_length, HEAD_DIM)
        """
        assert len(x.shape) == 3
        assert len(cache_key.shape) == 4
        assert len(cache_value.shape) == 4
        
        
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

        k = self.positional_encoder(k, offset=cache_key.shape[1])
        
        k = torch.cat((cache_key, k), dim=1)
        v = torch.cat((cache_value, v), dim=1)

        k, v = k.transpose(1, 2), v.transpose(1, 2)
        
        return q, k, v
        