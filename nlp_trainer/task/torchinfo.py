import torch
import torchinfo
from nlp_trainer.component.model.titans.mac import MAC


def main():
    with torch.no_grad():
        model = MAC(
            hidden_dim=768,
            num_heads=12,
            ff_dim=3072,
            persistent_memory_length=128,
            num_blocks=12,
            vocab_size=32000,
            memory_num_layers=2,
            temperature=1.0,
        ).to("cuda:0")
        torchinfo.summary(
            model, input_data=torch.randint(0, 32000, (2, 1024), device="cuda:0")
        )

        data = model(torch.randint(0, 32000, (2, 1024), device="cuda:0"))
        print(data[0].shape)
        print(data[0])


if __name__ == "__main__":
    main()
