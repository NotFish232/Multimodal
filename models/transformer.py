from transformers import BertModel, BertTokenizer
import torch as T
from torch import nn
from PIL import Image
from typing_extensions import Self


HUGGINGFACE_MODEL = "bert-base-uncased"


class TransformerModel(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(HUGGINGFACE_MODEL)
        self.model = BertModel.from_pretrained(HUGGINGFACE_MODEL)

    def forward(self: Self, text: str) -> T.Tensor:
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        input_ids = T.tensor(inputs["input_ids"], device=self.model.device).unsqueeze(0)
        token_type_ids = T.tensor(inputs["token_type_ids"], device=self.model.device).unsqueeze(0)
        attention_mask = T.tensor(inputs["attention_mask"], device=self.model.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return outputs


m = TransformerModel()
print(sum(i.numel() for i in m.parameters()))
print([t.shape for t in m("this is a test sentence with more things now").__dict__.values() if  isinstance(t, T.Tensor)])
