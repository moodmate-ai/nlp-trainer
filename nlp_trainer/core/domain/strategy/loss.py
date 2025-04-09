from abc import ABC, abstractmethod
from typing import TypeVar, Generic


BATCH = TypeVar("BATCH")
MODEL_OUTPUT = TypeVar("MODEL_OUTPUT")
LOSS_INPUT = TypeVar("LOSS_INPUT")


class LossInputStrategy(ABC, Generic[BATCH, MODEL_OUTPUT, LOSS_INPUT]):
    """
    How to make input of loss function?!
    """

    @abstractmethod
    def execute(self, batch: BATCH, model_output: MODEL_OUTPUT) -> LOSS_INPUT:
        pass
