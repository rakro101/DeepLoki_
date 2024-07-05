import os

import numpy as np
import pandas as pd

haul_pic_path = "data/loki_raw_output/0010_PS121-010-03/"
import shutil
from pathlib import Path

import pytorch_lightning as pl

from dtl_model import DtlModel
from loki_datasets import LokiDataModule, LokiPredictDataset, LokiTrainValDataset
from typing import Optional


def folder_name(confi: float, pred: str, treshold: float) -> str:
    """
    Return the folder_name, unknow if confidence is below treshold
    Args:
        confi: confidence score
        pred: prediction
        treshold: treshold

    Returns:

    """
    if confi > treshold:
        return pred
    else:
        return "Unknow"


def predict_folder(
        haul_pic_path: str = haul_pic_path,
        ending: str = ".bmp",
        arch: str = "dtl_resnet18_classifier",
        target: Optional[str] = None,
        tr: float = 0.5,
) -> pd.DataFrame:
    """
    predict all classes for the images in the folder
    Args:
        haul_pic_path: path to the images
        ending: file extension
        arch: classifier

    Returns:
        dataframe with predictions
    """
    dm = LokiDataModule(batch_size=1512, ending=ending, pred_data_path=haul_pic_path)
    pred_loader = dm.predict_dataloader()
    lrvd = LokiTrainValDataset()
    num_classes = lrvd.n_classes
    label_encoder = lrvd.label_encoder
    model = DtlModel(
        input_shape=(3, 300, 300),
        label_encoder=label_encoder,
        num_classes=num_classes,
        arch=arch,
        transfer=False,
        num_train_layers=1,
        learning_rate=0.0001,
    )
    trainer = pl.Trainer(
        max_epochs=5, accelerator="mps", devices="auto", deterministic=True
    )
    a = trainer.predict(model, pred_loader)
    names = [
        item for d in a for item in d["file_names"]
    ]  # item for d in a for item in d['outputs']
    preds = [item for d in a for item in d["preds"]]
    confis = [item for d in a for item in d["confis"]]
    results = pd.DataFrame()
    results["file_names"] = names
    results["preds"] = preds
    results["confis"] = confis
    results["folder"] = results.apply(
        lambda x: folder_name(x["confis"], x.preds, tr), axis=1
    )
    if target is None:
        results.to_csv(f"inference/csv/inference_results_{arch}.csv", sep=";")
    return results


def create_folder(path: str):
    """
    create folder
    Args:
        path: path

    Returns:

    """
    # creating a new directory called pythondirectory
    Path(path).mkdir(parents=True, exist_ok=True)
    return None


def create_folders(results: pd.DataFrame, target="inference/sorted"):
    """
    Create folder for sorting
    Args:
        results: dataframe
        target: file to folders

    Returns:

    """
    class_folders = np.unique(results["folder"])
    for cl in class_folders:
        temp_path = f"{target}/{cl}"
        create_folder(path=temp_path)
    return None


def copy_to_folder(results: pd.DataFrame, target="inference/sorted"):
    """
    copy the image to folder accordently to folder col
    Args:
        results: dataframe
        target: path

    Returns:

    """
    for row in results.iterrows():
        source = f"{row[1][0]}"
        dest = f"{target}/{row[1][3]}"
        filename = os.path.basename(source)
        # Use shutil.copyfile to copy the file from the original path to the destination directory
        # change always to png images
        shutil.copyfile(
            source,
            os.path.join(dest, filename.replace("." + filename.split(".")[-1], ".png")),
        )

    results.to_csv(f"{target.replace('sorted', 'csv')}_inference_results.csv", sep=";")
    return None


def main(
        haul_pic_path: str = haul_pic_path,
        ending: str = ".bmp",
        arch: str = "dino_resnet18_classifier",
        target: str = "inference/sorted",
        tr: float = 0.05,

):
    """
    main methods
    Args:
        haul_pic_path: path to the image
        ending: file extension of the images
        arch: classifier
        target: path to the sorted image folder

    Returns:

    """
    # get preds
    results = predict_folder(
        haul_pic_path=haul_pic_path, ending=ending, arch=arch, target=target, tr=tr,
    )
    # create folders
    create_folders(results, target)
    # copy to folders
    copy_to_folder(results, target)
    print("done")
    return None


if __name__ == "__main__":
    # main(haul_pic_path="data/loki_raw_output/0010_PS121-010-03/")
    # data/data_set_004/test/Bubble
    # main(haul_pic_path="data/data_set_004/test/Bubble", ending=".png", arch="dtl_resnet18_classifier")
    main(
        haul_pic_path="data/loki_raw_output/0010_PS121-010-03/",
        ending=".bmp",
        arch="dino_resnet18_classifier",
        target="inference/sorted",
    )
