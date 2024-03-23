from pathlib import Path
import pandas as pd
import gzip
import json


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

                name = file.with_suffix("").name
                file_to_labels[name] = labels.split(",")

        with open(out_file, "wt+") as f:
            json.dump(file_to_labels, f)


def main() -> None:
    unzip_all_data()
    find_all_labels()


if __name__ == "__main__":
    main()
