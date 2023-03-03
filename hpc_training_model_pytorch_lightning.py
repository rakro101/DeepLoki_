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
import sys
from pytorch_lightning.callbacks import EarlyStopping



torch.manual_seed(42)
seed = seed_everything(42, workers=True)

only_cols =['object_lat',
       'object_lon',
       'object_time',
       'object_depth_min',
       'object_depth_max',
       'object_bottom_depth',
       'object_pressure',
       'object_temperature',
       'object_salinity',
       'object_conductivity',
       'object_oxygen_concentration',
       'object_temperature_oxsens',
       'object_oxygen_saturation',
       'object_area_px',
       'object_form',
       'object_area',
       'object_length',
       'object_width',
       'object_convexity',
       'object_structure',
       'object_graymean',
       'object_kurtosis',
       'object_skewness',
       'object_hu_moment_1',
       'object_hu_moment_2',
       'object_hu_moment_3',
       'object_hu_moment_4',
       'object_hu_moment_5',
       'object_hu_moment_6',
       'object_hu_moment_7',
       'object_fourier_descriptor_01',
       'object_fourier_descriptor_02',
       'object_fourier_descriptor_03',
       'object_fourier_descriptor_04',
       'object_fourier_descriptor_05',
       'object_fourier_descriptor_06',
       'object_fourier_descriptor_07',
       'object_fourier_descriptor_08',
       'object_fourier_descriptor_09',
       'object_fourier_descriptor_10',
       'object_posx',
       'object_posy',
       'object_milliseconds',
       'object_timestamp',
       'object_lenght',
       'object_chlorophyll_a',
       'object_light',
       'object_speed',
       'object_dr._haardt_fluorescence_channel_a',
       'object_dr._haardt_fluorescence_channel_b',
       'object_dr._haardt_fluorescence_channel_c',
       'object_dr._haardt_fluorescence_channel_d',
            'object_mc_width',
            'object_mc_height',
            'object_mc_bx',
            'object_mc_by',
            'object_mc_circ.',
            'object_mc_area_exc',
            'object_mc_area',
            'object_mc_%area',
            'object_mc_major',
            'object_mc_minor',
            'object_mc_y',
            'object_mc_x',
            'object_mc_convex_area',
            'object_mc_min',
            'object_mc_max',
            'object_mc_mean',
            'object_mc_intden',
            'object_mc_perim.',
            'object_mc_elongation',
            'object_mc_range',
            'object_mc_perimareaexc',
            'object_mc_perimmajor',
            'object_mc_circex',
            'object_mc_angle',
            'object_mc_bounding_box_area',
            'object_mc_eccentricity',
            'object_mc_equivalent_diameter',
            'object_mc_euler_number',
            'object_mc_extent',
            'object_mc_local_centroid_col',
            'object_mc_local_centroid_row',
            'object_mc_solidity',
            ]

class LokiDataset(Dataset):
    def __init__(self, img_transform=None, feat_transform=None, target_transform=None):
        self.df_abt = pd.read_csv("output/allcruises_df_validated_5with_zoomie.csv")
        self.df_abt = self.df_abt.drop(['count','object_annotation_category', 'object_annotation_category_id'],axis=1)
        self.label_encoder = preprocessing.LabelEncoder()
        self.imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        # self.imputer_num.set_output(transform="pandas")
        self.numeric_columns = self.df_abt.select_dtypes(include='number').columns
        self.numeric_columns = list(set(self.numeric_columns).intersection(set(only_cols)))
        self.image_root = self.df_abt['root_path'].values
        self.image_path = self.df_abt['img_file_name'].values
        self.scaler = StandardScaler()
        self.scaler.fit(self.imputer_num.fit_transform(self.df_abt[self.numeric_columns]))
        self.features = torch.Tensor(self.scaler.transform(self.imputer_num.fit_transform(self.df_abt[self.numeric_columns])))
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

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # build dataset
        dataset = LokiDataset()
        # split dataset
        noi = int(0.1*len(dataset))
        self.train, self.val ,self.test = random_split(dataset,[int(7*noi), int(1.5*noi), int(1.5*noi), len(dataset)-int(7*noi)-2*int(1.5*noi)])[0:3]

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
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, transfer=False, num_train_layers=3, arch="resnet18"):
        super().__init__()

        # log hyperparameters

        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        self.num_train_layers = num_train_layers
        self.arch = arch
        # transfer learning if pretrained=True
        # print cuda
        self.print_cuda()
        if arch == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=transfer)
        elif arch == "vgg16":
            self.feature_extractor = models.vgg16(pretrained=transfer)
        elif arch == "vitb":
            self.feature_extractor = models.vit_b_16(pretrained=transfer)
        #self.feature_extractor = models.efficientnet_v2_l(pretrained=transfer)
        self.save_hyperparameters()
        if transfer:
            max_Children = int(len([child for child in self.feature_extractor.children()]))
            ct = max_Children
            for child in self.feature_extractor.children():
                ct -= 1
                if ct < self.num_train_layers:
                    for param in child.parameters():
                        param.requires_grad = True

        n_sizes = self._get_conv_output(input_shape)


        self.feature_dim = 43
        self.classifier = nn.Linear(n_sizes+self.feature_dim, num_classes)
        self.feature_part = nn.Sequential(
          nn.Linear(77,50),#nn.Linear(91,50),
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
        #self.accuracy = Accuracy()
        self.accuracy = Accuracy()

    # print cuda
    def print_cuda(self):
        ''' Print out GPU Details and Cuda Version '''
        try:
            self.log("cuda avaibable", torch.cuda.is_available())
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

        self.accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", self.accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, gt = batch[0], batch[1], batch[2]
        self.log("val/shape", batch_x.shape[0])
        out = self.forward(batch_x, batch_y)
        loss = self.criterion(out, gt)

        self.log("val/loss", loss)

        self.accuracy(out, gt)
        self.log("val/acc",self.accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, gt = batch[0], batch[1], batch[2]
        self.log("test/shape", batch_x.shape[0])
        out = self.forward(batch_x, batch_y)
        loss = self.criterion(out, gt)

        return {"loss": loss, "outputs": out, "gt": gt}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        output = torch.cat([x['outputs'] for x in outputs], dim=0)

        gts = torch.cat([x['gt'] for x in outputs], dim=0)

        self.log("test/loss", loss)
        self.accuracy(output, gts)
        self.log("test/acc", self.accuracy)

        self.test_gts = gts
        self.test_output = output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    bs_fit = False
    lr_fit = False
    dm = LokiDataModule(batch_size=16)
    model = LitModel((3, 224, 224), 43, transfer=True, num_train_layers=1, learning_rate=0.00052, arch="vitb")
    #trainer = pl.Trainer(max_epochs=25, accelerator="mps", devices="auto", deterministic=True)
    if bs_fit:
        trainer=pl.Trainer(max_epochs=1, accelerator="mps", devices="auto", deterministic=True, auto_scale_batch_size=True)
        trainer.tune(model, dm)
    elif lr_fit:
        trainer=pl.Trainer(max_epochs=1, accelerator="mps", devices="auto", deterministic=True, auto_lr_find=True)
        trainer.tune(model,dm)
    else:
        trainer = pl.Trainer(max_epochs=20, accelerator="mps", devices="auto", deterministic=True, callbacks=[EarlyStopping("val/loss", patience=5, mode="min")])
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)



