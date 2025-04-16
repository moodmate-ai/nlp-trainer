import torch
import torch.nn as nn
from nlp_trainer.component.model.titans.block.mac import MACBlock
from typing import Optional, TypedDict
from nlp_trainer.core.domain.port.model import NLPModel
from nlp_trainer.core.domain.entity.model import NLPModelType


class MACInput(TypedDict):
    x: torch.Tensor
    cache_key: Optional[list[torch.Tensor]] = None
    cache_value: Optional[list[torch.Tensor]] = None


class MACOutput(TypedDict):
    x: torch.Tensor
    cache_key: list[torch.Tensor]
    cache_value: list[torch.Tensor]
    mem_losses: list[torch.Tensor]


class MAC(NLPModel[MACInput, MACOutput], nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        persistent_memory_length: int,
        memory_num_layers: int,
        num_blocks: int,
        vocab_size: int,
        temperature: float = 1.0,
    ):
        super(MAC, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                MACBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    persistent_memory_length=persistent_memory_length,
                    memory_num_layers=memory_num_layers,
                    temperature=temperature,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_proj = nn.Linear(hidden_dim, vocab_size)

    def get_empty_cache(
        self, batch_size: int, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = torch.empty(batch_size, 0, self.num_heads, self.head_dim, device=device)
        return cache

    def forward(
        self,
        x: torch.Tensor,
        cache_key: Optional[list[torch.Tensor]] = None,
        cache_value: Optional[list[torch.Tensor]] = None,
    ):
        x = self.embedding(x)

        if cache_key is None or cache_value is None:
            cache_key = [
                self.get_empty_cache(x.shape[0], x.device)
                for _ in range(len(self.blocks))
            ]
            cache_value = [
                self.get_empty_cache(x.shape[0], x.device)
                for _ in range(len(self.blocks))
            ]

        new_cache_key = []
        new_cache_value = []
        mem_losses = []

        for block, key, value in zip(self.blocks, cache_key, cache_value):
            x, attn_k, attn_v, mem_loss = block(
                x, attention_cache_key=key, attention_cache_value=value
            )

            new_cache_key.append(attn_k)
            new_cache_value.append(attn_v)
            mem_losses.append(mem_loss)

        x = self.final_proj(x)

        return x, new_cache_key, new_cache_value, mem_losses

    def get_type(self) -> NLPModelType:
        return NLPModelType.TITANS_MAC

    def train_step(self, batch: MACInput) -> MACOutput:
        x, cache_key, cache_value, mem_losses = self.forward(
            batch["x"], batch.get("cache_key", None), batch.get("cache_value", None)
        )
        return MACOutput(
            x=x, cache_key=cache_key, cache_value=cache_value, mem_losses=mem_losses
        )

    def predict_step(self, batch: MACInput) -> MACOutput:
        x, cache_key, cache_value, mem_losses = self.forward(
            batch["x"], batch.get("cache_key", None), batch.get("cache_value", None)
        )
        return MACOutput(
            x=x, cache_key=cache_key, cache_value=cache_value, mem_losses=mem_losses
        )
