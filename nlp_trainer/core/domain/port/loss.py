from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import torch
from nlp_trainer.core.domain.entity.loss import LossFunctionType

T = TypeVar("T")


class LossFunction(Generic[T], ABC):
    @abstractmethod
    def get_type(self) -> LossFunctionType:
        pass

    @abstractmethod
    def calculate_loss(self, batch: T) -> torch.Tensor:
        pass
