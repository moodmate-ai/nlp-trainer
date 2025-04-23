from nlp_trainer.di.app import Application
from nlp_trainer.core.domain.usecase.train import TrainUsecase
from nlp_trainer.component.dataloader.huggingface import HuggingFaceDataLoader
from nlp_trainer.component.tokenizer.llama import LlamaTokenizer
from nlp_trainer.component.loss.perplexity import PerplexityLoss
from nlp_trainer.component.loss.zero import ZeroLoss
from nlp_trainer.component.model.titans.mac import MAC
from .registry import LanguageModelingRegistry
from typing import TypedDict
import logging
import torch


logger = logging.getLogger(__name__)


class LanguageModelingConfig(TypedDict):
    max_epoch: int


class LanguageModelingApplication(Application):
    def __init__(self, config: LanguageModelingConfig):
        super().__init__()
        self.config = config
        self.dataloader = HuggingFaceDataLoader(
            "HuggingFaceFW/fineweb-edu",
            "CC-MAIN-2024-10",
            "train",
            streaming=True,
            batch_size=5,
            num_workers=4,
            pin_memory=True,
        ).get_batch_iterator()

        self.registry = LanguageModelingRegistry(
            device="cuda:0", tokenizer=LlamaTokenizer()
        )

        self.model = MAC(
            hidden_dim=512,
            num_heads=8,
            ff_dim=512,
            persistent_memory_length=32,
            memory_num_layers=2,
            num_blocks=12,
            vocab_size=32000,
        ).to("cuda:0")

        self.loss_fn_list = [PerplexityLoss().to("cuda:0"), ZeroLoss()]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-4)

        self.usecase = TrainUsecase(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn_list=self.loss_fn_list,
            train_registry=self.registry,
        )

    def run(self):
        ## Log config
        logger.info({"language modeling config": self.config})

        for epoch in range(0, self.config["max_epoch"]):
            self.usecase.execute(epoch, self.dataloader)


if __name__ == "__main__":
    config = LanguageModelingConfig(max_epoch=10)
    app = LanguageModelingApplication(config=config)
    app.run()
