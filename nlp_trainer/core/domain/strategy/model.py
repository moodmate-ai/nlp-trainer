from abc import ABC, abstractmethod
from typing import TypeVar, Generic


BATCH = TypeVar("BATCH")
MODEL_INPUT = TypeVar("MODEL_INPUT")


class ModelInputStrategy(ABC, Generic[BATCH, MODEL_INPUT]):
    @abstractmethod
    def execute(self, batch: BATCH) -> MODEL_INPUT:
        pass
