from pathlib import Path
import gzip

MIMIC_PATH = "./data/mimiciv/2.2"


def unzip_all_data() -> None:
    for file in Path(MIMIC_PATH).glob("**/*"):
        if file.is_file() and file.suffixes == [".csv", ".gz"]:
            directory = file.parent
            new_directory = Path("./processed_data") / directory.relative_to("./data")
            new_file = new_directory / file.with_suffix("").name

            if new_file.exists():
                continue
            
            Path(new_directory).mkdir(exist_ok=True, parents=True)
            with gzip.open(file, "rb") as zip_f:
                file_content = zip_f.read()
                with open(new_file, "wb+") as f:
                    f.write(file_content)


unzip_all_data()
