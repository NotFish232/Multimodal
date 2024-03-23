from pathlib import Path
import pandas as pd
import gzip


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

def get_all_labels() -> "dict[str, list[str]]":
    labels = {}
    for file in Path(PROCESSED_DATA_PATH).glob("**/*"):
        if file.is_file() and file.suffix == ".csv":
            df = pd.read_csv(file)
            name = file.with_suffix("").name
            labels[name] = [c for c in df.columns]
    
    return labels


unzip_all_data()
print(get_all_labels())
