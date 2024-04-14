import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self
from transformers import BertModel, ConvNextV2Model


class MimicCXRModel(nn.Module):
    def __init__(self: Self, vocab_size: int) -> None:
        super().__init__()

        self.conv_net = ConvNextV2Model.from_pretrained(
            "facebook/convnextv2-base-22k-384"
        )
        self.ftc1 = nn.Linear(1024, 512)
        self.ftc2 = nn.Linear(512, 256)

        self.transformer = BertModel.from_pretrained("bert-base-uncased")

        self.final_layer = nn.Linear(768, vocab_size)

    def forward(
        self: Self,
        tokens: T.Tensor,
        attention_mask: T.Tensor,
        images: T.Tensor,
        num_images: T.Tensor,
    ) -> T.Tensor:
        image_features = self.conv_net(images).pooler_output
        image_features = F.leaky_relu(self.ftc1(image_features))

        batched_features_list = []
        images_idx = 0
        for num_image in num_images:
            num_image = num_image.item()

            batched_features = image_features[images_idx : images_idx + num_image]
            batched_features = T.mean(batched_features, dim=0, keepdim=True)

            batched_features_list.append(batched_features)

            images_idx += num_image
            
        batched_image_features = T.concat(batched_features_list)
        batched_image_features = F.leaky_relu(self.ftc2(batched_image_features))
        batched_image_features = batched_image_features.to(T.int64)

        complete_embeddings = T.concat((tokens, batched_image_features), dim=1)
        attention_mask = T.concat((attention_mask, T.ones_like(attention_mask)), dim=1)

        transformer_output = self.transformer(
            complete_embeddings, attention_mask=attention_mask
        ).last_hidden_state

        final_output = self.final_layer(transformer_output)

        return final_output
