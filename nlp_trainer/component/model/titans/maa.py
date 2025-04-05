import torch
import torch.nn as nn
from nlp_trainer.component.model.embedding.rope import RotaryPositionalEmbedding
from nlp_trainer.component.model.titans.module.attention import MultiHeadAttention
from nlp_trainer.component.model.titans.module.memory import LongTermMemory
from nlp_trainer.component.model.norm.layer import LayerNorm


class MAABlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        persistent_memory_length: int,
    ):
        super(MAABlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.persistent_memory = nn.Parameter(
            torch.randn(1, persistent_memory_length, hidden_dim)
        )
        self.memory = LongTermMemory(hidden_dim)

        self.positional_encoder = RotaryPositionalEmbedding(
            self.hidden_dim // self.num_heads,
            max_seq_len=persistent_memory_length + 12800,
            theta=10000,
        )

        self.attention = MultiHeadAttention(
            num_heads, hidden_dim, self.positional_encoder
        )

        self.layer_norm1 = LayerNorm(hidden_dim)
        self.layer_norm2 = LayerNorm(hidden_dim)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ff_dim, hidden_dim),
        )

    def get_empty_cache(self, batch_size: int, device):
        return self.attention.get_empty_cache(batch_size=batch_size, device=device)

    def forward(
        self, x: torch.Tensor, attention_cache: tuple[torch.Tensor, torch.Tensor] = None
    ):
        assert len(x.shape) == 3
        skip = x
        x = self.layer_norm1(x)
        ## Retrieve memories

        persistent_memory = self.persistent_memory.repeat(x.shape[0], 1, 1)

        # x = x.view(x.shape[0], x.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        # memory_query = self.positional_encoder(x)
        # q, k, v = self.attention.get_qkv(x)
        memory_output = self.memory(x=x, is_update=False)

        x = torch.cat((persistent_memory, memory_output, x), dim=1)

        x, _, cache = self.attention(
            x=x,
            memory_length=persistent_memory.shape[1] + memory_output.shape[1],
            cache=attention_cache,
        )

        x = x[:, persistent_memory.shape[1] + memory_output.shape[1] :, :]
        x = x + skip
        skip = x

        x = self.layer_norm2(x)
        v = self.memory(x=x, is_update=True)
        x = x + v

        x = self.feed_forward(x)
        x = skip + x

        return x, cache


class MAA(nn.Module):
    """
    Memory As Attention Store

    query -> memory -> value

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        persistent_memory_length: int,
        num_blocks: int,
        vocab_size: int,
    ):
        super(MAA, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                MAABlock(hidden_dim, num_heads, ff_dim, persistent_memory_length)
                for _ in range(num_blocks)
            ]
        )

        self.final_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, attention_cache: tuple[torch.Tensor, torch.Tensor] = None
    ):
        if attention_cache is None:
            attention_cache = [
                block.get_empty_cache(x.shape[0], x.device) for block in self.blocks
            ]

        x = self.embedding(x)

        new_cache = []

        for block, cache in zip(self.blocks, attention_cache):
            x, c = block(x, attention_cache=cache)
            new_cache.append(c)

        x = self.final_proj(x)
        return x, new_cache


if __name__ == "__main__":
    import torchinfo

    model = MAA(
        hidden_dim=1280,
        num_heads=20,
        ff_dim=5120,
        persistent_memory_length=256,
        num_blocks=36,
        vocab_size=32000,
    )
    torchinfo.summary(model, input_data=torch.randint(0, 32000, (2, 20)))
