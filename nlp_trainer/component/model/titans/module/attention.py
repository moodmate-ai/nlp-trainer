import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        positional_encoder: nn.Module,
    ):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)

        self.positional_encoder = positional_encoder

        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def get_empty_cache(
        self, batch_size: int, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty(
            batch_size, 0, self.num_heads, self.head_dim, device=device
        ), torch.empty(batch_size, 0, self.num_heads, self.head_dim, device=device)

    def forward(
        self, x, memory_length: int, cache: tuple[torch.Tensor, torch.Tensor] = None
    ):
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

        if cache is not None:
            cache_k, cache_v = cache
            k = self.positional_encoder(k, offset=cache_k.shape[1])

            k = torch.cat((cache_k, k), dim=1)
            v = torch.cat((cache_v, v), dim=1)

            cache = (k, v)

        k, v = k.transpose(1, 2), v.transpose(1, 2)

        ## 3. attention
        attention_weights = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        ## 4. mask
        mask = torch.zeros_like(attention_weights)
        mask[:, :, :memory_length, memory_length:] = 1
        mask[:, :, memory_length:, memory_length:] = torch.triu(
            torch.ones_like(mask[:, :, memory_length:, memory_length:]), diagonal=1
        )

        attention_weights = attention_weights.masked_fill(mask == 1, -float("inf"))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        attention = (attention_weights @ v).transpose(1, 2)

        attention = attention.contiguous().view(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        attention = self.linear(attention)

        return attention, attention_weights, cache
