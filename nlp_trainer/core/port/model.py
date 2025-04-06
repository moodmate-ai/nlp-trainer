from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from nlp_trainer.core.entity.model import NLPModelType

T = TypeVar("T")
E = TypeVar("E")


class NLPModel(Generic[T, E], ABC):
    @abstractmethod
    def get_model_type(self) -> NLPModelType:
        pass

    @abstractmethod
    def train_step(self, batch: T) -> E:
        pass

    @abstractmethod
    def predict_step(self, batch: T) -> E:
        pass
