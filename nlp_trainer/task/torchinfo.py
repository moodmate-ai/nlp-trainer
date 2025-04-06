import torch
import torchinfo
from nlp_trainer.component.model.titans.mac import MAC


def main():
    model = MAC(
        hidden_dim=1024,
        num_heads=16,
        ff_dim=4096,
        persistent_memory_length=256,
        num_blocks=24,
        vocab_size=32000,
    ).to("cuda:0")
    torchinfo.summary(
        model, input_data=torch.randint(0, 32000, (2, 1024), device="cuda:0")
    )


if __name__ == "__main__":
    main()
