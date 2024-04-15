from models import MimicCXRModel
from dataset import MimicCXRDataset

import torch as T


def main() -> None:
    device = T.device("cuda:5" if T.cuda.is_available() else "cpu")

    dataset = MimicCXRDataset(device)

    model = MimicCXRModel(len(dataset.tokenizer)).to(device)
    model.load_state_dict(T.load("saved_models/model.pt"))
    model.eval()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    batch = dataset[3800]

    information_tokens = batch["information_tokens"].unsqueeze(0)
    information_attention_mask = batch["information_attention_mask"].unsqueeze(0)
    images = batch["images"]
    num_images = batch["num_images"].unsqueeze(0)
    result_tokens = batch["result_tokens"]

    with T.no_grad():
        output = model(
            information_tokens, information_attention_mask, images, num_images
        )

    print(dataset.tokenizer.decode(T.argmax(output, dim=-1)[0]))
    print(dataset.tokenizer.decode(result_tokens))


if __name__ == "__main__":
    main()
