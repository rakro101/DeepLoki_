import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models


import wandb
wandb.login()

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image

class LokiDataset(Dataset):
    def __init__(self, img_transform=None, feat_transform=None, target_transform=None):
        self.df_abt = pd.read_csv("output/allcruises_df_validated.csv")
        self.df_abt = self.df_abt.drop(['count','object_annotation_category', 'object_annotation_category_id'],axis=1)
        self.label_encoder = preprocessing.LabelEncoder()
        self.imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.imputer_num.set_output(transform="pandas")
        numeric_columns = self.df_abt.select_dtypes(include='number').columns
        self.image_root = self.df_abt['root_path'].values
        self.image_path = self.df_abt['img_file_name'].values
        self.features = torch.Tensor(self.imputer_num.fit_transform(self.df_abt[numeric_columns]).values)
        self.label_encoder.fit(self.df_abt['label'])
        self.label = torch.Tensor(self.label_encoder.transform(self.df_abt['label'])).type(torch.LongTensor)
        self.img_transform = img_transform
        self.feat_transform = feat_transform
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
        if self.feat_transform:
            feature = self.feat_transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return image, feature, label

class LokiDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.ToTensor(),
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              #transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              #transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.num_classes = 23

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # build dataset
        dataset = LokiDataset()
        # split dataset
        noi = 2000 # int(0.1*len(dataset))
        self.train, self.val ,self.test = random_split(dataset,[7*noi, int(1.5*noi), int(1.5*noi), len(dataset)-10*noi])[0:3]

        self.train.dataset.img_transform = self.augmentation
        self.val.dataset.img_transform = self.transform
        self.test.dataset.img_transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=10)

class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, transfer=False):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        # print cuda
        self.print_cuda()
        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=transfer)
        #self.feature_extractor = models.efficientnet_v2_l(pretrained=transfer)

        if transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        n_sizes = self._get_conv_output(input_shape)


        self.feature_dim = 43
        self.classifier = nn.Linear(n_sizes+self.feature_dim, num_classes)
        self.feature_part = nn.Sequential(
          nn.Linear(91,50),
          nn.BatchNorm1d(num_features=50),
          nn.Dropout(p=0.05),
          nn.ReLU(),
          #
          nn.Linear(50, 50),
          nn.Dropout(p=0.05),
          nn.BatchNorm1d(num_features=50),
          nn.ReLU(),
          #
          nn.Linear(50, 50),
          nn.Dropout(p=0.05),
          nn.BatchNorm1d(num_features=50),
          nn.ReLU(),
          #
          nn.Linear(50, self.feature_dim),
          nn.Dropout(p=0.05),
          nn.BatchNorm1d(num_features=self.feature_dim),
          nn.ReLU(),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    # print cuda
    def print_cuda(self):
        ''' Print out GPU Details and Cuda Version '''
        try:
            self.log(f"cuda avaibable {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.log('__Python VERSION:', sys.version)
                self.log('__pyTorch VERSION:', torch.__version__)
                self.log('__CUDA VERSION:', torch.version.cuda)
                self.log('__Number CUDA Devices:', torch.cuda.device_count())
                self.log('__Devices:')
                from subprocess import call
                call(["nvidia-smi", "--format=csv",
                      "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
                self.log('Active CUDA Device: GPU', torch.cuda.current_device())
                self.log('Available devices ', torch.cuda.device_count())
                self.log('Current cuda device ', torch.cuda.current_device())
        except Exception as err:
            print(err)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    # will be used during inference
    def forward(self, x, y):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       y = self.feature_part(y)
       x = torch.cat((x, y), dim=1)
       x = self.classifier(x)

       return x

    def training_step(self, batch):
        batch_x, batch_y, gt = batch[0], batch[1], batch[2]
        out = self.forward(batch_x, batch_y)
        loss = self.criterion(out, gt)

        acc = self.accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, gt = batch[0], batch[1], batch[2]
        out = self.forward(batch_x, batch_y)
        loss = self.criterion(out, gt)

        self.log("val/loss", loss)

        acc = self.accuracy(out, gt)
        self.log("val/acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, gt = batch[0], batch[1], batch[2]
        out = self.forward(batch_x, batch_y)
        loss = self.criterion(out, gt)

        return {"loss": loss, "outputs": out, "gt": gt}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        output = torch.cat([x['outputs'] for x in outputs], dim=0)

        gts = torch.cat([x['gt'] for x in outputs], dim=0)

        self.log("test/loss", loss)
        acc = self.accuracy(output, gts)
        self.log("test/acc", acc)

        self.test_gts = gts
        self.test_output = output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    dm = LokiDataModule(batch_size=256)
    model = LitModel((3, 300, 300), 43, transfer=True)
    trainer = pl.Trainer(logger=WandbLogger(project="TransferLearning"), max_epochs=50, accelerator="cpu")
    trainer.fit(model, dm)
    trainer.test(model, dm)
    wandb.finish()
