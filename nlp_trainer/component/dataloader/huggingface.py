from nlp_trainer.core.domain.entity.dataloader import Dataloader
from datasets import load_dataset
from torch.utils import data
from typing import Iterable, Any
from nlp_trainer.core.domain.entity.dataloader import DataloaderType
# use name="sample-10BT" to use the 10BT sample


class HuggingFaceDataLoader(Dataloader):
    def __init__(
        self,
        path: str,
        name: str,
        split: str,
        streaming: bool,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int,
        persistent_workers: bool,
        timeout: int,
    ):
        dataset = load_dataset(path, name, split=split, streaming=streaming)
        self.dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            timeout=timeout,
        )

    def get_type(self) -> DataloaderType:
        return DataloaderType.HUGGINGFACE

    def get_batch_iterator(self) -> Iterable[Any]:
        return self.dataloader


if __name__ == "__main__":
    fw = HuggingFaceDataLoader(
        "HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-10", "train", streaming=True
    )
    for i in fw:
        print(i)
        break
