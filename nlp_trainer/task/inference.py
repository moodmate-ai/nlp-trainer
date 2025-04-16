import torch
from nlp_trainer.component.model.titans.mac import MAC, MACInput
from nlp_trainer.component.tokenizer.llama import LlamaTokenizer


def main():
    tokenizer = LlamaTokenizer()
    state_dict = torch.load("model.pt", weights_only=True)
    model = MAC(
        hidden_dim=768,
        num_heads=12,
        ff_dim=3072,
        persistent_memory_length=128,
        memory_num_layers=2,
        num_blocks=12,
        vocab_size=32000,
        temperature=1.0,
    ).to("cuda:0")
    model.load_state_dict(state_dict)

    with torch.no_grad():
        x = tokenizer.encode("Parents are, ")
        x = torch.tensor([x], dtype=torch.long, device="cuda:0")
        mac_input = MACInput(x=x)
        mac_output = model.predict_step(mac_input)

        y = mac_output["x"]
        y = y[0].tolist()
        result = []
        for ll in y:
            m = -100000.0
            w = 0
            for idx, num in enumerate(ll):
                if num > m:
                    w = idx
            result.append(w)

        y = tokenizer.decode(result)

        print(y)


if __name__ == "__main__":
    main()
