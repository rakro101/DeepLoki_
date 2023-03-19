import pytorch_lightning as pl
from torch import nn
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
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

torch.manual_seed(42)
seed = seed_everything(42, workers=True)

class LokiTrainValDataset(Dataset):
    def __init__(self, img_transform=None, target_transform=None):
        #self.df_abt = pd.read_csv("output/allcruises_df_validated_5with_zoomie.csv")
        self.df_abt = pd.read_csv('output/update_allcruises_df_validated_5with_zoomie_20230303.csv')
        self.df_abt = self.df_abt[self.df_abt["object_cruise"] != "PS99.2"]
        self.df_abt = self.df_abt[self.df_abt["label"] != "Artefact"]# remove artefact
        self.df_abt = self.df_abt.drop(['count','object_annotation_category', 'object_annotation_category_id','new_index'],axis=1)
        # num features
        self.numeric_columns = self.df_abt.select_dtypes(include='number').columns
        self.numeric_columns = self.df_abt[self.numeric_columns].dropna(axis =1, how='all').columns
        self.imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.imputer_num.set_output(transform="pandas")
        print('cols feature train val', self.df_abt[self.numeric_columns].shape)
        self.features = torch.Tensor(self.imputer_num.fit_transform(self.df_abt[self.numeric_columns]).values)
        self.label_encoder = preprocessing.LabelEncoder()
        self.image_root = self.df_abt['root_path'].values
        self.image_path = self.df_abt['img_file_name'].values
        self.label_encoder.fit(self.df_abt['label'])
        self.label = torch.Tensor(self.label_encoder.transform(self.df_abt['label'])).type(torch.LongTensor)
        self.n_classes = len(np.unique(self.label))
        self.img_transform = img_transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.df_abt)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_root[item], self.image_path[item])
        image = Image.open(img_path).convert('RGB')
        feature = self.features[item]
        label = self.label[item]
        if self.img_transform:
            image = self.img_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, feature, label

class LokiTestDataset(Dataset):
    def __init__(self, img_transform=None, target_transform=None, label_encoder=None, numeric_columns = None, imputer_num = None,):
        self.df_abt = pd.read_csv("output/update_wo_artefacts_test_dataset_PS992_03032023.csv")
        self.df_abt = self.df_abt[self.df_abt["label"] != "Artefact"]  # remove artefact
        self.df_abt = self.df_abt.drop(['count','object_annotation_category', 'object_annotation_category_id'],axis=1)
        self.label_encoder = label_encoder
        self.numeric_columns = numeric_columns
        self.imputer_num = imputer_num
        print('cols feature test', self.df_abt[self.numeric_columns].shape)
        self.features = torch.Tensor(self.imputer_num.transform(self.df_abt[self.numeric_columns]).values)
        self.image_root = self.df_abt['root_path'].values
        self.image_path = self.df_abt['img_file_name'].values
        self.label = torch.Tensor(self.label_encoder.transform(self.df_abt['label'])).type(torch.LongTensor)
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_abt)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_root[item], self.image_path[item])
        image = Image.open(img_path).convert('RGB')
        feature = self.features[item]
        label = self.label[item]
        if self.img_transform:
            image = self.img_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,  feature, label


class LokiDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # build dataset
        train_val_dataset = LokiTrainValDataset()
        encoder_train = train_val_dataset.label_encoder
        numeric_columns = train_val_dataset.numeric_columns
        imputer_num = train_val_dataset.imputer_num
        test_dataset = LokiTestDataset(label_encoder=encoder_train, numeric_columns=numeric_columns, imputer_num=imputer_num)
        # split dataset
        number_of_samples =len(train_val_dataset)
        n_train_samples =int(0.8*number_of_samples)
        n_val_samples =int(0.2*number_of_samples)
        n_rest =number_of_samples -n_train_samples -n_val_samples
        self.train, self.val = random_split(train_val_dataset,[n_train_samples, n_val_samples, n_rest])[0:2]
        self.test = random_split(test_dataset, [len(test_dataset), 0])[0]



        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.ToTensor(),
              transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
              transforms.RandomAutocontrast(p=0.25),
              transforms.RandomPerspective(distortion_scale=0.25, p=0.25),
              transforms.RandomAdjustSharpness(sharpness_factor=4, p=0.25),
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Resize(size=224),
              transforms.CenterCrop(size=224),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.train.dataset.img_transform = self.augmentation
        self.val.dataset.img_transform = self.transform
        self.test.dataset.img_transform = self.transform
    def prepare_data(self):
        pass




    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=10)

