from nlp_trainer.core.domain.port.loss import LossFunction
from nlp_trainer.core.domain.entity.loss import LossFunctionType
from typing import TypedDict
import torch
from nlp_trainer.component.loss.perplexity import PerplexityLoss


class CompositeLossInput(TypedDict):
    model_output: list[str]
    loss_input: list[str]


class CompositeLoss(LossFunction):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.perplexity_loss = PerplexityLoss()

    def calculate_loss(self, batch: CompositeLossInput) -> torch.Tensor:
        perplexity_loss = self.perplexity_loss.calculate_loss(batch["loss_input"])
        return perplexity_loss

    def get_loss_function_type(self) -> LossFunctionType:
        return LossFunctionType.CUSTOM
