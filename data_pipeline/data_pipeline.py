import gzip
from pathlib import Path
from PIL import Image
import numpy as np
import zipfile
import json
import pydicom
from tqdm import tqdm

DATA_PATH = "./data"
MIMIC_PATH = f"{DATA_PATH}/physionet.org/files/mimic-cxr/2.0.0"
PROCESSED_DATA_PATH = "./processed_data"


def unzip_all_data() -> None:
    for file in Path(MIMIC_PATH).glob("*.csv.gz"):
        directory = file.parent
        new_directory = Path(PROCESSED_DATA_PATH) / directory.relative_to(MIMIC_PATH)
        new_file = new_directory / file.with_suffix("").name

        if new_file.exists():
            continue

        Path(new_directory).mkdir(exist_ok=True, parents=True)
        with gzip.open(file, "rb") as zip_f:
            file_content = zip_f.read()
            with open(new_file, "wb+") as f:
                f.write(file_content)

    for file in Path(MIMIC_PATH).glob("*.zip"):
        directory = file.parent
        new_directory = (
            Path(PROCESSED_DATA_PATH)
            / directory.relative_to(MIMIC_PATH)
            / file.with_suffix("").name
        )

        if new_directory.exists():
            continue

        Path(new_directory).mkdir(exist_ok=True, parents=True)

        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(new_directory)


def parse_image_and_report_data() -> None:
    path_to_image_data = Path(MIMIC_PATH) / "files"
    path_to_processed_data = Path(PROCESSED_DATA_PATH) / "dataset"
    for patient_dir in tqdm(sorted(path_to_image_data.glob("*/*"))):
        if not patient_dir.is_dir():
            continue

        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            out_path = path_to_processed_data / f"{patient_dir.name}_{study_dir.name}"

            if out_path.exists():
                continue

            out_path.mkdir(parents=True)

            try:
                dcm_file = next(study_dir.glob("*.dcm"))
                ds = pydicom.read_file(dcm_file)  # type: ignore
                pixels = (255 * (ds.pixel_array / ds.pixel_array.max())).astype(
                    np.uint8
                )
                image = Image.fromarray(pixels)
                image = image.resize((1024, 1024))
                image.save(out_path / "image.png")

                info_file = study_dir.with_suffix(".txt")

                info = info_file.read_text()
                findings_idx = info.find("FINDINGS:")
                findings_len = len("FINDINGS:")
                impression_idx = info.find("IMPRESSION:")
                impression_len = len("IMPRESSION:")

                findings = info[findings_idx + findings_len : impression_idx].strip()
                impression = info[impression_idx + impression_len :].strip()

                info_json = {"findings": findings, "impression": impression}
                json.dump(info_json, open(out_path / "info.json", "w+"))

            except Exception as e:
                print(f"Processing {patient_dir.name}_{study_dir.name}: {e}")
                raise e


def main() -> None:
    unzip_all_data()
    parse_image_and_report_data()


if __name__ == "__main__":
    main()
