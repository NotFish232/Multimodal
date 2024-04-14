from transformers import AutoImageProcessor, ConvNextV2Model
import torch as T
from torch import nn
from PIL import Image
from typing_extensions import Self

image = Image.open(
    "processed_data/dataset/p10000032_s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.png"
).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-atto-1k-224")
model = ConvNextV2Model.from_pretrained("facebook/convnextv2-atto-1k-224")
print(sum(i.numel() for i in model.parameters()))
inputs = image_processor(image, return_tensors="pt")

with T.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print([v.shape for v in outputs.values() if isinstance(v, T.Tensor)])

HUGGINGFACE_MODEL = "facebook/convnextv2-nano-22k-384"

class CNNModel(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.model = ConvNextV2Model.from_pretrained("facebook/convnextv2-nano-22k-384")
        self.processor = AutoImageProcessor.from_pretrained(HUGGINGFACE_MODEL)
    
    def forward(self: Self, image: Image.Image) -> T.Tensor:
        inputs = self.processor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs
