from nlp_trainer.core.domain.entity.loss import LossFunctionType
import torch
from nlp_trainer.component.loss.perplexity import PerplexityLoss, PerplexityLossInput
from nlp_trainer.component.loss.zero import ZeroLoss, ZeroLossInput
from nlp_trainer.core.domain.registry.train import TrainRegistry
from nlp_trainer.component.dataloader.huggingface import HuggingFaceDataLoader
from nlp_trainer.core.domain.strategy.model import ModelInputStrategy
from nlp_trainer.core.domain.strategy.loss import LossInputStrategy
from nlp_trainer.component.tokenizer.llama import LlamaTokenizer
from nlp_trainer.component.model.titans.mac import MAC, MACInput, MACOutput
from nlp_trainer.core.domain.usecase.train import TrainUsecase


class MACInputStrategy(ModelInputStrategy):
    def __init__(
        self,
        max_seq_len: int,
        device: str,
        tokenizer: LlamaTokenizer,
    ):
        super(MACInputStrategy, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    def execute(self, batch) -> MACInput:
        batch = batch["text"]
        x = []

        for item in batch:
            item = self.tokenizer.encode(item)
            if len(item) > self.max_seq_len:
                item = item[: self.max_seq_len]

            x.append(item)

        min_seq_len = min(len(item) for item in x)
        x = [item[:min_seq_len] for item in x]

        x = torch.tensor(x, dtype=torch.long, device=self.device)
        return MACInput(x=x)


class PerplexityStrategy(LossInputStrategy):
    def __init__(self, device: str, tokenizer: LlamaTokenizer, max_seq_len: int):
        super(PerplexityStrategy, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def execute(
        self, batch: torch.Tensor, model_output: MACOutput
    ) -> PerplexityLossInput:
        batch = batch["text"]
        x = []

        for item in batch:
            item = self.tokenizer.encode(item)
            if len(item) > self.max_seq_len:
                item = item[: self.max_seq_len]

            x.append(item)

        min_seq_len = min(len(item) for item in x)
        x = [item[:min_seq_len] for item in x]

        x = torch.tensor(x, dtype=torch.long, device=self.device)

        return PerplexityLossInput(y_pred=model_output["x"], y_true=x)


class MemoryLossStrategy(LossInputStrategy):
    def __init__(self, device: str):
        super(MemoryLossStrategy, self).__init__()
        self.device = device

    def execute(self, batch: torch.Tensor, model_output: MACOutput) -> ZeroLossInput:
        mem_losses = model_output["mem_losses"]
        mem_losses = torch.tensor(
            mem_losses, dtype=torch.float32, device=self.device
        ).sum()
        return ZeroLossInput(y_pred=mem_losses)


class LanguageModelingRegistry(TrainRegistry):
    def __init__(self, device: str, tokenizer: LlamaTokenizer):
        super(LanguageModelingRegistry, self).__init__()
        self._model_input_strategy = MACInputStrategy(
            max_seq_len=768, device=device, tokenizer=tokenizer
        )
        self._loss_input_strategy = {
            LossFunctionType.PERPLEXITY: PerplexityStrategy(
                device=device, tokenizer=tokenizer, max_seq_len=768
            ),
            LossFunctionType.ZERO: MemoryLossStrategy(device=device),
        }

    def get_model_input_strategy(self) -> ModelInputStrategy:
        return self._model_input_strategy

    def get_loss_input_strategy(
        self, loss_fn_type: LossFunctionType
    ) -> LossInputStrategy:
        return self._loss_input_strategy[loss_fn_type]


def main():
    dataloader = HuggingFaceDataLoader(
        "HuggingFaceFW/fineweb-edu",
        "CC-MAIN-2024-10",
        "train",
        streaming=True,
        batch_size=5,
        num_workers=4,
        pin_memory=True,
    ).get_batch_iterator()

    registry = LanguageModelingRegistry(device="cuda:0", tokenizer=LlamaTokenizer())
    model = MAC(
        hidden_dim=512,
        num_heads=8,
        ff_dim=512,
        persistent_memory_length=32,
        memory_num_layers=2,
        num_blocks=12,
        vocab_size=32000,
    ).to("cuda:0")

    loss_fn_list = [PerplexityLoss().to("cuda:0"), ZeroLoss()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    usecase = TrainUsecase(
        model=model,
        optimizer=optimizer,
        loss_fn_list=loss_fn_list,
        train_registry=registry,
    )

    for epoch in range(10):
        usecase.execute(dataloader)


if __name__ == "__main__":
    main()
