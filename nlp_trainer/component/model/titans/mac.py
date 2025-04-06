import torch
import torch.nn as nn
from nlp_trainer.component.model.module.embedding.rope import RotaryPositionalEmbedding
from nlp_trainer.component.model.module.attention.multihead import MultiHeadAttention
from nlp_trainer.component.model.module.attention.qkv import MultiHeadQKVCreator
from nlp_trainer.component.model.module.memory.mlp import TitansMlpMemory
from nlp_trainer.component.model.module.norm.layer import LayerNorm
from typing import Optional

class MACBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        persistent_memory_length: int
    ):
        super(MACBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.persistent_memory = nn.Parameter(
            torch.randn(1, persistent_memory_length, hidden_dim)
        )
        self.memory = TitansMlpMemory(
            num_layers=2,
            hidden_dim = hidden_dim,
            ff_dim=hidden_dim*2
        )

        self.positional_encoder =  RotaryPositionalEmbedding(
            hidden_dim // num_heads,
            max_seq_len=persistent_memory_length + 12800,
            theta=10000,
        )
        
        self.qkv_creator = MultiHeadQKVCreator(
            num_heads, hidden_dim, self.positional_encoder
        )

        self.attention = MultiHeadAttention(
            num_heads, hidden_dim
        )

        self.layer_norm1 = LayerNorm(hidden_dim)
        self.layer_norm2 = LayerNorm(hidden_dim)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ff_dim, hidden_dim),
        )
        
    def _retrieve_memories(self, x: torch.Tensor):
        persistent_memory = self.persistent_memory.repeat(x.shape[0], 1, 1)
        memory_output = self.memory(x=x.detach())
        
        return persistent_memory, memory_output
    
    def _calculate_attention(
        self, 
        x: torch.Tensor, 
        attention_cache_key: torch.Tensor, 
        attention_cache_value: torch.Tensor,
        memory_length: int
    ):
        q, k, v = self.qkv_creator(
            x, attention_cache_key, attention_cache_value
        )
        ## shape: [batch, num_head, mem + seq, head_dim]
        
        attention_cache_key = k.transpose(1, 2)
        attention_cache_value = v.transpose(1, 2)
        
        x, _ = self.attention(
            q = q,
            k = k,
            v = v,
            memory_length = memory_length
        )
        ## shape: [batch, mem + seq, hidden dim]
        
        return x, attention_cache_key, attention_cache_value

    def forward(
        self, 
        x: torch.Tensor, 
        attention_cache_key: torch.Tensor,
        attention_cache_value: torch.Tensor
    ):
        assert len(x.shape) == 3
        skip = x
        x = self.layer_norm1(x)
        
        ## Retrieve memories
        ##
        persistent_memory, memory_output = self._retrieve_memories(x)
        memory_length = persistent_memory.shape[1] + memory_output.shape[1]

        x = torch.cat((persistent_memory, memory_output, x), dim=1)

        ## Calculate attention
        ##
        x, attention_cache_key, attention_cache_value = self._calculate_attention(
            x = x,
            attention_cache_key = attention_cache_key,
            attention_cache_value = attention_cache_value,
            memory_length = memory_length
        )
        ## shape: [batch, mem + seq, hidden dim]
        
        ## Calculate memory loss (surprise metric)
        ##
        memory_answer = x[:, persistent_memory.shape[1] : memory_length, :]
        memory_loss = torch.norm((memory_output - memory_answer.detach()), p=2, dim=-1).sum()
        
        ## Feed forward
        ##
        x = x[:, memory_length :, :] + skip
        skip = x

        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = skip + x

        return x, attention_cache_key, attention_cache_value, memory_loss


class MAC(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        persistent_memory_length: int,
        num_blocks: int,
        vocab_size: int,
    ):
        super(MAC, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                MACBlock(hidden_dim, num_heads, ff_dim, persistent_memory_length)
                for _ in range(num_blocks)
            ]
        )

        self.final_proj = nn.Linear(hidden_dim, vocab_size)
        
    def get_empty_cache(
        self, batch_size: int, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        cache = torch.empty(
            batch_size, 0, self.num_heads, self.head_dim, device=device
        )
        return cache

        
    def forward(
        self, 
        x: torch.Tensor,
        cache_key: Optional[list[torch.Tensor]] = None,
        cache_value: Optional[list[torch.Tensor]] = None
    ):
        x = self.embedding(x)

        if cache_key is None or cache_value is None:
            cache_key = [self.get_empty_cache(x.shape[0], x.device) for _ in range(len(self.blocks))]
            cache_value = [self.get_empty_cache(x.shape[0], x.device) for _ in range(len(self.blocks))]

        new_cache_key = []
        new_cache_value = []
        mem_losses = []
        
        for block, key, value in zip(self.blocks, cache_key, cache_value):
            x, attn_k, attn_v, mem_loss = block(x, attention_cache_key=key, attention_cache_value=value)
            
            new_cache_key.append(attn_k)
            new_cache_value.append(attn_v)
            mem_losses.append(mem_loss)

        x = self.final_proj(x)
        
        return x, new_cache_key, new_cache_value, mem_losses
    
    
if __name__ == "__main__":
    import torchinfo

    model = MAC(
        hidden_dim=1024,
        num_heads=16,
        ff_dim=4096,
        persistent_memory_length=256,
        num_blocks=24,
        vocab_size=32000,
    ).to("cuda:0")
    torchinfo.summary(model, input_data=torch.randint(0, 32000, (5, 1024), device="cuda:0"))
