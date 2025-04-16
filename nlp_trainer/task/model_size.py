import torch
from nlp_trainer.component.model.titans.mac import MAC, MACInput
from nlp_trainer.component.loss.perplexity import PerplexityLoss, PerplexityLossInput
import time


def main():
    # model = MAC(
    #     hidden_dim=768,
    #     num_heads=12,
    #     ff_dim=3072,
    #     persistent_memory_length=128,
    #     memory_num_layers=2,
    #     num_blocks=12,
    #     vocab_size=32000,
    #     temperature=1.0,
    # ).to("cuda:0")
    model = MAC(
        hidden_dim=512,
        num_heads=8,
        ff_dim=1024,
        persistent_memory_length=32,
        num_blocks=12,
        vocab_size=32000,
        memory_num_layers=2,
        temperature=1.0,
    ).to("cuda:0")

    loss_function = PerplexityLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    max_iteration = 10000
    for idx in range(max_iteration):
        time_start = time.time()
        optimizer.zero_grad()

        x = torch.randint(0, 32000, (5, 768), device="cuda:0")

        mac_input = MACInput(x=x)
        mac_output = model.train_step(mac_input)

        loss_input = PerplexityLossInput(y_pred=mac_output["x"], y_true=x)
        loss = loss_function.calculate_loss(loss_input)

        loss_log = loss.to("cpu").detach().numpy()

        loss.backward()
        optimizer.step()

        time_end = time.time()
        print(f"{idx}: {time_end - time_start} seconds, loss: {loss_log}")

        if idx == max_iteration:
            torch.save(model.state_dict(), "model.pt")
            break


if __name__ == "__main__":
    main()
