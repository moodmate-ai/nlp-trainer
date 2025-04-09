from abc import ABC, abstractmethod
from typing import TypeVar, Generic


MODEL_OUTPUT = TypeVar("MODEL_OUTPUT")
LOSS_INPUT = TypeVar("LOSS_INPUT")


class ModelToLossStrategy(ABC, Generic[MODEL_OUTPUT, LOSS_INPUT]):
    @abstractmethod
    def execute(self, model_output: MODEL_OUTPUT) -> LOSS_INPUT:
        pass
