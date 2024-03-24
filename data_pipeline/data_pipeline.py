import gzip
import json
from pathlib import Path
import pandas as pd

DATA_PATH = "./data"
MIMIC_PATH = f"{DATA_PATH}/mimiciv/2.2"
PROCESSED_DATA_PATH = "./processed_data"


def unzip_all_data() -> None:
    for file in Path(MIMIC_PATH).glob("**/*"):
        if file.is_file() and file.suffixes == [".csv", ".gz"]:
            directory = file.parent
            new_directory = Path(PROCESSED_DATA_PATH) / directory.relative_to(DATA_PATH)
            new_file = new_directory / file.with_suffix("").name

            if new_file.exists():
                continue

            Path(new_directory).mkdir(exist_ok=True, parents=True)
            with gzip.open(file, "rb") as zip_f:
                file_content = zip_f.read()
                with open(new_file, "wb+") as f:
                    f.write(file_content)


def find_all_labels() -> None:
    out_file = Path(PROCESSED_DATA_PATH) / "labels.json"

    if not out_file.exists():
        file_to_labels = {}
        for file in Path(PROCESSED_DATA_PATH).glob("**/*"):
            if file.is_file() and file.suffix == ".csv":
                with open(file, "rt") as f:
                    labels = f.readline().strip()

                file_to_labels[str(file)] = labels.split(",")

        with open(out_file, "wt+") as f:
            json.dump(file_to_labels, f)


def find_all_icd_codes() -> None:
    out_file = Path(PROCESSED_DATA_PATH) / "icd_codes.json"

    if not out_file.exists():
        labels_file = Path(PROCESSED_DATA_PATH) / "labels.json"
        with open(labels_file, "rt") as f:
            file_to_labels = json.load(f)
        
        icd_codes = set()

        for file, labels in file_to_labels.items():
            for label in labels:
                if label == "icd_code":
                    df = pd.read_csv(file)
                    icd_codes |= {*df[label].unique()}
        
        with open(out_file, "wt+") as f:
            json.dump([*icd_codes], f)
   

def main() -> None:
    unzip_all_data()
    find_all_labels()
    find_all_icd_codes()

if __name__ == "__main__":
    main()
