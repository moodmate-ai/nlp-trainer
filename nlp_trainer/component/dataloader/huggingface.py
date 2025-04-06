from datasets import load_dataset
from torch.utils import data
# use name="sample-10BT" to use the 10BT sample

class HuggingFaceDataLoader(data.DataLoader):
    def __init__(
        self, 
        path: str, 
        name: str, 
        split: str, 
        streaming: bool = True,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        dataset = load_dataset(path, name, split=split, streaming=streaming)
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    fw = HuggingFaceDataLoader("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-10", "train", streaming=True)
    for i in fw:
        print(i)
        break
