from models import MimicCXRModel
from dataset import MimicCXRDataset, custom_collate_fn

import torch as T
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

NUM_EPOCHS = 100
BATCH_SIZE = 8
LR = 5e-4


def main() -> None:
    device = T.device("cuda:5" if T.cuda.is_available() else "cpu")

    dataset = MimicCXRDataset(device)
    dataloader = DataLoader(dataset, BATCH_SIZE, collate_fn=custom_collate_fn)

    model = MimicCXRModel(len(dataset.tokenizer)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), LR)

    writer = SummaryWriter()

    for epoch in range(1, NUM_EPOCHS + 1):
        acc_loss = 0.0
        for batch in tqdm(dataloader):
            information_tokens = batch["information_tokens"]
            information_attention_mask = batch["information_attention_mask"]
            images = batch["images"]
            num_images = batch["num_images"]
            result_tokens = batch["result_tokens"]

            output = model(
                information_tokens, information_attention_mask, images, num_images
            )

            output_reshaped = output.view(-1, output.size(-1))
            result_reshaped = result_tokens.view(-1)

            loss = criterion(output_reshaped, result_reshaped)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            acc_loss += loss.item()
        acc_loss /= len(dataloader)

        print(f"Loss: {acc_loss:.4f}")

        writer.add_scalar("loss/train", acc_loss, epoch)
        writer.flush()

        if epoch % 10 == 0:
            T.save(model.state_dict(), "saved_models/model.pt")


if __name__ == "__main__":
    main()
