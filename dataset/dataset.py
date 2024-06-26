import torch as T
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoImageProcessor  # type: ignore
from PIL import Image
from typing_extensions import Self
from typing import List
from pathlib import Path
import json

DATA_DIR = "./processed_data/dataset"


class MimicCXRDataset(Dataset):
    def __init__(self: Self, device: T.device) -> None:
        self.data_dirs = sorted(Path(DATA_DIR).iterdir())

        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/convnextv2-tiny-22k-384",
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            "google/bert_uncased_L-4_H-256_A-4"
        )

        self.device = device

    def __len__(self: Self) -> int:
        return len(self.data_dirs)

    def __getitem__(self: Self, idx: int) -> dict:
        data_dir = self.data_dirs[idx]

        info_json = json.load(open(data_dir / "info.json"))
        information = info_json["information"]
        result = info_json["result"]

        image_paths = sorted(data_dir.glob("*.png"))
        images = [Image.open(p).convert("RGB") for p in image_paths]
        num_images = T.tensor(len(images), device=self.device)
        processed_images = self.processor(images, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)

        information_tokens = self.tokenizer.encode_plus(
            information,
            None,
            max_length=256,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        information_input_ids = T.tensor(
            information_tokens["input_ids"], device=self.device
        )
        information_attention_mask = T.tensor(
            information_tokens["attention_mask"], device=self.device
        )

        result_tokens = self.tokenizer.encode_plus(
            result,
            None,
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        result_input_ids = T.tensor(result_tokens["input_ids"], device=self.device)

        return {
            "information_tokens": information_input_ids,
            "information_attention_mask": information_attention_mask,
            "images": processed_images,
            "num_images": num_images,
            "result_tokens": result_input_ids,
        }


def custom_collate_fn(batch: "List[dict]") -> dict:
    result: dict = {}

    for entry in batch:
        for key, val in entry.items():
            if key not in result:
                result[key] = []
            result[key].append(val)

    for key, val in result.items():
        result[key] = T.vstack(val)

    return result
