import os
import numpy as np
import pandas as pd
haul_pic_path = "data/loki_raw_output/0010_PS121-010-03/"
from loki_datasets import LokiDataModule, LokiTrainValDataset, LokiPredictDataset
from dtl_model import DtlModel
import pytorch_lightning as pl
from pathlib import Path
import shutil

def folder_name(confi, pred, treshold):
    if confi > treshold:
        return pred
    else:
        return "Unknow"

def predict_folder(haul_pic_path=haul_pic_path, ending=".bmp", arch="dtl_resnet18_classifier"):
    dm = LokiDataModule(batch_size=256, ending=ending, pred_data_path=haul_pic_path)
    pred_loader = dm.predict_dataloader()
    lrvd = LokiTrainValDataset()
    num_classes = lrvd.n_classes
    label_encoder = lrvd.label_encoder
    model = DtlModel(input_shape=(3,300,300), label_encoder=label_encoder, num_classes=num_classes, arch=arch, transfer=False, num_train_layers=1, learning_rate=0.0001)
    trainer = pl.Trainer(max_epochs=5, accelerator="mps", devices="auto", deterministic=True)
    a = trainer.predict(model, pred_loader)
    names = [item for d in a for item in d['file_names']]# item for d in a for item in d['outputs']
    preds = [item for d in a for item in d['preds']]
    confis = [item for d in a for item in d['confis']]
    results = pd.DataFrame()
    results["file_names"]=names
    results["preds"]=preds
    results["confis"]=confis
    results["folder"]=results.apply(lambda x: folder_name(x["confis"], x.preds, 0.2), axis=1)
    results.to_csv("inference/csv/inference_results.csv", sep=";")
    return results

def create_folder(path):
    # creating a new directory called pythondirectory
    Path(path).mkdir(parents=True, exist_ok=True)
    return None

def create_folders(results, target="inference/sorted"):
    class_folders = np.unique(results["folder"])
    for cl in class_folders:
        temp_path = f"{target}/{cl}"
        create_folder(path=temp_path)
    return None

def copy_to_folder(results, target="inference/sorted"):
    for row in results.iterrows():
        source =f'{row[1][0]}'
        dest = f'{target}/{row[1][3]}'
        filename = os.path.basename(source)
        # Use shutil.copyfile to copy the file from the original path to the destination directory
        shutil.copyfile(source, os.path.join(dest, filename))
    return None

def main(haul_pic_path=haul_pic_path, ending=".bmp", arch="dino_resnet18_classifier", target="inference/sorted"):
    # get preds
    results = predict_folder(haul_pic_path=haul_pic_path,ending=ending, arch=arch)
    # create folders
    create_folders(results, target)
    # copy to folders
    copy_to_folder(results, target)
    print('done')

if __name__ == "__main__":
    #main(haul_pic_path="data/loki_raw_output/0010_PS121-010-03/")
    #data/data_set_004/test/Bubble
    #main(haul_pic_path="data/data_set_004/test/Bubble", ending=".png", arch="dtl_resnet18_classifier")
    main(haul_pic_path="data/loki_raw_output/0010_PS121-010-03/", ending=".bmp", arch="dino_resnet18_classifier", target="inference/sorted")


