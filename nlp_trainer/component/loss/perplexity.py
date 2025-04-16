import torch
import torch.nn as nn
from nlp_trainer.core.domain.port.loss import LossFunction
from nlp_trainer.core.domain.entity.loss import LossFunctionType
from typing import TypedDict


class PerplexityLossInput(TypedDict):
    y_pred: torch.Tensor
    y_true: torch.Tensor


class PerplexityLoss(LossFunction[PerplexityLossInput], nn.Module):
    def __init__(self):
        super(PerplexityLoss, self).__init__()

        self.loss_fct = nn.CrossEntropyLoss()

    def get_type(self) -> LossFunctionType:
        return LossFunctionType.PERPLEXITY

    def calculate_loss(self, batch: PerplexityLossInput) -> torch.Tensor:
        logits = batch["y_pred"]
        labels = batch["y_true"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        perplexity = torch.exp(loss)

        return perplexity
