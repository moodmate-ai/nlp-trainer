from nlp_trainer.core.domain.port.loss import LossFunction
from nlp_trainer.core.domain.entity.loss import LossFunctionType
import torch
from typing import TypedDict


class MSELossInput(TypedDict):
    y_pred: torch.Tensor
    y_true: torch.Tensor


class MSELoss(LossFunction):
    def __init__(self):
        super(MSELoss, self).__init__()

    def get_type(self) -> LossFunctionType:
        return LossFunctionType.MSE

    def calculate_loss(self, batch: MSELossInput) -> torch.Tensor:
        return torch.sum((batch["y_pred"] - batch["y_true"]) ** 2)
