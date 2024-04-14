from transformers import AutoImageProcessor, ConvNextV2Model
import torch as T
from torch import nn
from PIL import Image
from typing_extensions import Self


HUGGINGFACE_MODEL = "facebook/convnextv2-nano-22k-384"


class CNNModel(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(HUGGINGFACE_MODEL)
        self.model = ConvNextV2Model.from_pretrained(HUGGINGFACE_MODEL)

    def forward(self: Self, image: Image.Image) -> T.Tensor:
        inputs = self.processor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs
