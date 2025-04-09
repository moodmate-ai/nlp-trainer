from abc import ABC, abstractmethod
from nlp_trainer.core.domain.entity.dataloader import DataloaderType
from typing import TypeVar, Generic, Iterator


BATCH = TypeVar("BATCH")


class Dataloader(ABC, Generic[BATCH]):
    @abstractmethod
    def get_type(self) -> DataloaderType:
        pass

    @abstractmethod
    def get_batch_iterator(self, *args, **kwargs) -> Iterator[BATCH]:
        pass
