from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
import torchvision.models as models
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import seed_everything
from time import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

torch.manual_seed(42)
seed = seed_everything(42, workers=True)

t0 = time()
from loki_datasets import LokiDataModule, LokiTrainValDataset
from dtl_model import DtlModel

if __name__ == '__main__':
    test_loader = LokiDataModule(batch_size=1024).test_dataloader()
    tv_dl =LokiTrainValDataset(Dataset)
    num_classes = tv_dl.n_classes
    print(num_classes)
    le = tv_dl.label_encoder
    model = DtlModel(input_shape=(3,300,300), num_classes=num_classes)
    #c_path ="lightning_logs/version_60/checkpoints/epoch=2-step=912.ckpt" #paperversion
    c_path = "lightning_logs/version_60/checkpoints/epoch=2-step=912.ckpt"
    checkpoint = torch.load(c_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    run_prefix = c_path.split("/")[1]+'new'
    print(run_prefix)



    # print(len(test))

    gt = []
    pred = []
    feature_test =[]
    sm = nn.Softmax(dim=1)


    dl_model = True
    if dl_model:
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                print(i)
                temp_pred = model(image)
                temp_pred = sm(temp_pred).argmax(dim=1)
                temp_pred = le.inverse_transform([l.item() for l in temp_pred])
                pred.extend(temp_pred)
                label = le.inverse_transform([l.item() for l in label])
                gt.extend(label)



        print(len(gt))



        all_cls = list(np.unique(gt))
        print(all_cls)

        clsf_report = pd.DataFrame(classification_report(y_true = gt, y_pred = pred, output_dict=True)).transpose()
        clsf_report.to_csv(f"paper/tables/C4_{run_prefix}_lightning_class_report.csv", sep=";")
        print(clsf_report)



        cm = confusion_matrix(gt, pred)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_cls)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_cls).plot()
        fig_rf = disp.figure_
        fig_rf.set_figwidth(20)
        fig_rf.set_figheight(20)
        #fig_rf.suptitle('Plot of confusion matrix')
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(f'paper/figures/C4_{run_prefix}resnet_combon_model_sklearn_confusion_matrix.jpg')
        plt.show()


