from models import MimicCXRModel
from dataset import MimicCXRDataset, custom_collate_fn

import torch as T
from torch.utils.data import DataLoader

device = T.device("cuda" if T.cuda.is_available() else "cpu")

dataset = MimicCXRDataset(device)
dataloader = DataLoader(dataset, batch_size=5, collate_fn=custom_collate_fn)

model = MimicCXRModel(len(dataset.tokenizer)).to(device)

batch = next(iter(dataloader))
information_tokens = batch["information_tokens"]
information_attention_mask = batch["information_attention_mask"]
images = batch["images"]
num_images = batch["num_images"]

print(sum(p.numel() for p in model.parameters()))
x = model(information_tokens, information_attention_mask, images, num_images)[0]
x = T.argmax(x, dim=1)
print(dataset.tokenizer.decode(x))