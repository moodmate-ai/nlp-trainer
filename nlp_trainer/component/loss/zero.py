import torch
from nlp_trainer.core.domain.port.loss import LossFunction
from typing import TypedDict
from nlp_trainer.core.domain.entity.loss import LossFunctionType


class ZeroLossInput(TypedDict):
    y_pred: torch.Tensor


class ZeroLoss(LossFunction):
    """
    Loss function that returns the sum of the predicted values to be zero.
    """

    def __init__(self):
        super(ZeroLoss, self).__init__()

    def get_type(self) -> LossFunctionType:
        return LossFunctionType.ZERO

    def calculate_loss(self, batch: ZeroLossInput) -> torch.Tensor:
        y_pred = batch["y_pred"]
        y_pred = y_pred.sum()
        return y_pred
