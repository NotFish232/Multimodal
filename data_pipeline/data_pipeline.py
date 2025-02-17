import gzip
from pathlib import Path
import zipfile
from tqdm import tqdm
from itertools import chain
import pandas as pd
import shutil
import json
import math
from typing import List, Dict
import traceback

DATA_PATH = "./data/physionet.org/files"
PROCESSED_DATA_PATH = "./processed_data"

MIMIC_PATH = f"{DATA_PATH}/mimic-cxr/2.1.0"
MIMIC_JPG_PATH = f"{DATA_PATH}/mimic-cxr-jpg/2.1.0"


def process_data_files() -> None:
    out_dir = Path(PROCESSED_DATA_PATH) / "data_files"

    for file in chain(
        Path(MIMIC_PATH).glob("*.csv.gz"), Path(MIMIC_JPG_PATH).glob("*.csv.gz")
    ):
        directory = file.parent
        new_directory = out_dir / directory.relative_to(DATA_PATH)
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
            out_dir / directory.relative_to(DATA_PATH) / file.with_suffix("").name
        )

        if new_directory.exists():
            continue

        Path(new_directory).mkdir(exist_ok=True, parents=True)

        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(new_directory)

    for file in Path(MIMIC_JPG_PATH).glob("*.csv"):
        directory = file.parent
        new_directory = out_dir / directory.relative_to(DATA_PATH)
        new_file = new_directory / file.name

        if new_file.exists():
            continue

        Path(new_directory).mkdir(exist_ok=True, parents=True)
        shutil.copy(file, new_file)


def parse_image_and_report_data() -> None:
    images_path = Path(MIMIC_JPG_PATH) / "files"
    dataset_path = Path(PROCESSED_DATA_PATH) / "dataset"

    processed_mimic_path = f"{PROCESSED_DATA_PATH}/data_files/mimic-cxr/2.1.0"
    processed_mimic_jpg_path = f"{PROCESSED_DATA_PATH}/data_files/mimic-cxr-jpg/2.1.0"

    metadata_file = f"{processed_mimic_jpg_path}/mimic-cxr-2.0.0-metadata.csv"
    metadata_df = pd.read_csv(
        metadata_file, index_col=False
    )  # data dim does not match header dim

    record_file = f"{processed_mimic_path}/cxr-record-list.csv"
    record_df = pd.read_csv(record_file)

    chexpert_file = f"{processed_mimic_jpg_path}/mimic-cxr-2.0.0-chexpert.csv"
    chexpert_df = pd.read_csv(chexpert_file)
    negbio_file = f"{processed_mimic_jpg_path}/mimic-cxr-2.0.0-negbio.csv"
    negbio_df = pd.read_csv(negbio_file)
    radiologist_file = (
        f"{processed_mimic_jpg_path}/mimic-cxr-2.1.0-test-set-labeled.csv"
    )
    radiologist_df = pd.read_csv(radiologist_file)

    split_file = f"{processed_mimic_jpg_path}/mimic-cxr-2.0.0-split.csv"
    split_df = pd.read_csv(split_file)

    image_dirs = tqdm(sorted((f for f in images_path.glob("*/*/*") if f.is_dir())))

    dataset_path.mkdir(exist_ok=True)

    for image_dir in image_dirs:
        image_dirs.set_description(image_dir.parts[-2])  # type: ignore
        try:
            image_info: List[Dict[str, str | None]] = []

            image_files = sorted(image_dir.glob("*.jpg"))

            if len(image_files) == 0:
                continue

            for image_file in image_files:
                dicom_id = image_file.stem
                metadata = metadata_df[metadata_df["dicom_id"] == dicom_id]
                subject_id = metadata["subject_id"].values[0]
                study_id = metadata["study_id"].values[0]
                performed_procedure_step_description = metadata[
                    "PerformedProcedureStepDescription"
                ].values[0]
                view_position = metadata["ViewPosition"].values[0]

                image_path = record_df[record_df["dicom_id"] == dicom_id][
                    "path"
                ].values[0]

                # all the text paths should be the same
                text_path = str(Path(image_path).parent.with_suffix(".txt"))
                text_path = f"{processed_mimic_path}/mimic-cxr-reports/{text_path}"

                image_path = str(
                    Path(f"{MIMIC_JPG_PATH}/{image_path}").with_suffix(".jpg")
                )
                image_path = f"./{image_path}"

                image_info.append(
                    {
                        "dicom_id": dicom_id,
                        "image_path": image_path,
                        "view_position": (
                            None if str(view_position) == "nan" else str(view_position)
                        ),
                    }
                )

            chexpert_predictions = (
                chexpert_df[chexpert_df["study_id"] == study_id]
                .drop(["subject_id", "study_id"], axis=1)
                .iloc[0]
                .to_dict()
            )
            negbio_predictions = (
                negbio_df[negbio_df["study_id"] == study_id]
                .drop(["subject_id", "study_id"], axis=1)
                .iloc[0]
                .to_dict()
            )

            radiologist_predictions = None
            if study_id in radiologist_df["study_id"].values:
                radiologist_predictions = (
                    radiologist_df[radiologist_df["study_id"] == study_id]
                    .drop("study_id", axis=1)
                    .iloc[0]
                    .to_dict()
                )

            map_preds = lambda x: x and {
                k.replace(" ", "_").lower(): None if math.isnan(float(v)) else int(v)
                for k, v in x.items()
            }

            data = {
                "subject_id": int(subject_id),
                "study_id": int(study_id),
                "performed_procedure_step_description": (
                    None
                    if str(performed_procedure_step_description) == "nan"
                    else str(performed_procedure_step_description)
                ),
                "images": image_info,
                "text_path": text_path,
                "chexpert_predictions": map_preds(chexpert_predictions),
                "negbio_predictions": map_preds(negbio_predictions),
                "radiologist_predictions": map_preds(radiologist_predictions),
            }

            split_info = split_df[split_df["subject_id"] == subject_id]["split"].values[
                0
            ]

            (dataset_path / split_info).mkdir(exist_ok=True)

            filename = str(dataset_path / split_info / f"{subject_id}_{study_id}.json")

            json.dump(data, open(filename, "w+"), indent=4)

        except Exception as e:
            print(f"Error processsing: {image_dir}")
            print(traceback.format_exc())


def run_data_pipeline() -> None:
    # unzips / moves all csvs
    process_data_files()
    parse_image_and_report_data()


def main() -> None:
    run_data_pipeline()


if __name__ == "__main__":
    main()
