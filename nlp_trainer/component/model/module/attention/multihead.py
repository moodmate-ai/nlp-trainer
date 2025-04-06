import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int
    ):
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, 
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_length: int
    ):
        """
        Input: 
        - q: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        - k: [BATCH_SIZE, NUM_HEADS, CACHE_LEN + SEQ_LEN, HEAD_DIM]
        - v: [BATCH_SIZE, NUM_HEADS, CACHE_LEN + SEQ_LEN, HEAD_DIM]
        - memory_length:
            length of memory values, prefix in sequence
        
        
        1. Attention weights:
        [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM] @ [BATCH_SIZE, NUM_HEADS, HEAD_DIM, CACHE_LEN + SEQ_LEN]
        = [BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN + CACHE_LEN]
        
        
        2. Masking on attention weights in each head:
        - memory values cannot be compared with future information
            [ : memory_length , CACHE_LEN + memory_length : ]
        - upper triangle masking for other sequence(real input seq)
            [ memory_length : , CACHE_LEN + memory+length : ] -> upper triangle
        """
        assert len(q.shape) == 4
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = q.shape
        HIDDEN_DIM = NUM_HEADS * HEAD_DIM
        CACHE_LEN = k.shape[2] - SEQ_LEN
        
        
        ## 1. attention weights
        ## 
        attention_weights = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)


        ## 2. mask
        ##
        mask = torch.zeros_like(attention_weights)
        
        mask[:, :, :memory_length, CACHE_LEN + memory_length:] = 1
        
        mask[:, :, memory_length:, CACHE_LEN + memory_length:] = torch.triu(
            torch.ones_like(mask[:, :, memory_length:, CACHE_LEN + memory_length:]),
            diagonal=1
        )

        attention_weights = attention_weights.masked_fill(mask == 1, -float("inf"))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        # [b, head, seq, cache + seq]


        ## 3. Calculate attention
        ## 
        
        attention = (attention_weights @ v).transpose(1, 2)
        # [b, seq, head, head_dim]

        attention = attention.contiguous().view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        attention = self.linear(attention)

        return attention, mask