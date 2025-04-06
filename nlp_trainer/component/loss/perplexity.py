import torch
import torch.nn as nn
from nlp_trainer.core.port.loss import LossFunction
from nlp_trainer.core.entity.loss import LossFunctionType
from typing import TypedDict


class PerplexityLossInput(TypedDict):
    y_pred: torch.Tensor
    y_true: torch.Tensor


class PerplexityLoss(LossFunction[PerplexityLossInput], nn.Module):
    def __init__(self):
        super(PerplexityLoss, self).__init__()

    def get_loss_function_type(self) -> LossFunctionType:
        return LossFunctionType.PERPLEXITY

    def calculate_loss(self, batch: PerplexityLossInput) -> torch.Tensor:
        pass
