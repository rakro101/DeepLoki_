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

class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, transfer=False, num_train_layers=3, arch="resnet18"):
        super().__init__()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        self.num_train_layers = num_train_layers
        self.arch = arch
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
        self.classifier = nn.Linear(n_sizes, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()


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
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       return x

    def training_step(self, batch):
        batch_x, gt = batch[0], batch[1]
        out = self.forward(batch_x)
        loss = self.criterion(out, gt)

        self.accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", self.accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, gt = batch[0], batch[1]
        out = self.forward(batch_x)
        loss = self.criterion(out, gt)
        self.log("val/loss", loss)
        self.accuracy(out, gt)
        self.log("val/acc",self.accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, gt = batch[0], batch[1]
        out = self.forward(batch_x)
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

class LokiDataset(Dataset):
    def __init__(self, df = "output/allcruises_df_validated_5with_zoomie.csv", img_transform=None, target_transform=None):
        self.df_abt = pd.read_csv(df)
        #self.df_abt = self.df_abt[self.df_abt["object_cruise"] != "PS99.2"]
        #self.df_abt = self.df_abt[self.df_abt["label"] != "Artefact"]  # remove artefact
        self.df_abt = self.df_abt.drop(['count','object_annotation_category', 'object_annotation_category_id'],axis=1)
        self.label_encoder = preprocessing.LabelEncoder()
        self.image_root = self.df_abt['root_path'].values
        self.image_path = self.df_abt['img_file_name'].values
        self.label_encoder.fit(self.df_abt['label'])
        self.df_abt = self.df_abt[self.df_abt["label"] != "Artefact"]  # remove artefact
        self.label = torch.Tensor(self.label_encoder.transform(self.df_abt['label'])).type(torch.LongTensor)
        self.img_transform = img_transform
        self.target_transform = target_transform



    def __len__(self):
        return len(self.df_abt)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_root[item], self.image_path[item])
        image = Image.open(img_path).convert('RGB')

        label = self.label[item]
        if self.img_transform:
            image = self.img_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    model = LitModel(input_shape=(3,224,224), num_classes=43)
    c_path ="lightning_logs/version_47/checkpoints/epoch=6-step=3920.ckpt"
    checkpoint = torch.load(c_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    run_prefix = c_path.split("/")[1]+'new'
    print(run_prefix)

    img_transoform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Resize(size=224),
                  transforms.CenterCrop(size=224),
                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
    test_data = LokiDataset(img_transform=img_transoform, df="output/test_dataset_PS992.csv")
    # split dataset
    # print(len(test_data))

    test = random_split(test_data,[len(test_data)])[0]
    test_loader = DataLoader(test, batch_size=8196)

    # print(len(test))

    gt = []
    pred = []
    feature_test =[]
    sm = nn.Softmax(dim=1)
    le = LokiDataset().label_encoder
    lt =test_data.label_encoder
    dl_model = True
    if dl_model:
        with torch.no_grad():
            for i, (image, label) in enumerate(test_loader):
                print(i)
                temp_pred = model(image)
                temp_pred = sm(temp_pred).argmax(dim=1)
                temp_pred = le.inverse_transform([l.item() for l in temp_pred])
                pred.extend(temp_pred)
                label = lt.inverse_transform([l.item() for l in label])
                gt.extend(label)



        print(len(gt))



        all_cls = le.inverse_transform([r for r in range(0,43)])

        clsf_report = pd.DataFrame(classification_report(y_true = gt, y_pred = pred, output_dict=True)).transpose()
        clsf_report.to_csv(f"paper/tables/C4_{run_prefix}_lightning_class_report.csv", sep=";")
        print(clsf_report)


        try:
            cm = confusion_matrix(gt, pred)
            #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_cls)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            fig = disp.figure_
            fig.set_figwidth(20)
            fig.set_figheight(20)
            fig.xticks_rotation="vertical"
            fig.suptitle('Plot of confusion matrix')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            plt.savefig(f'paper/figures/C4_{run_prefix}resnet_combon_model_sklearn_confusion_matrix.jpg')
        except:
            pass

        plt.show()

        ###
    rf_model = True
    if rf_model:
        # train data loop
        test_dataset = pd.read_csv("output/test_dataset_PS992.csv")
        test_label = le.transform(test_dataset['label'])
        test_dataset = test_dataset.drop(['count','object_annotation_category', 'object_annotation_category_id', 'label'],axis=1)
        ####
        from sklearn.compose import ColumnTransformer
        from sklearn.datasets import fetch_openml
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        train_df_abt = pd.read_csv("output/allcruises_df_validated_5with_zoomie.csv")
        train_df_abt = train_df_abt[train_df_abt["object_cruise"] != "PS99.2"]
        train_label = le.transform(train_df_abt['label'])
        train_df_abt = train_df_abt.drop(['count','object_annotation_category', 'object_annotation_category_id', 'label'],axis=1)
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        from sklearn.compose import make_column_selector as selector

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, selector(dtype_include="number")),
                ("cat", categorical_transformer, selector(dtype_include="category")),
            ]
        )
        ###
        clf = Pipeline(
            steps=[("preprocessor", preprocessor),
                   ("classifier", RandomForestClassifier(max_depth=20, n_estimators=100, max_features=10))]
        )
        #clf = RandomForestClassifier(max_depth=20, n_estimators=100, max_features=10)#RandomForestClassifier(max_depth=100, random_state=42)

        clf.fit(train_df_abt, train_label)
        score = clf.score(test_dataset, test_label)
        preds = clf.predict(test_dataset)

        # rf class report
        rf_class_report = pd.DataFrame(classification_report(y_true = test_label, y_pred = preds , output_dict=True)).transpose()
        print(rf_class_report)
        print("rbf score:", score)
        rf_class_report.to_csv(f"paper/tables/4c_{run_prefix}_rf_sklearn_class_report.csv", sep=";")
        # rf save confusion matrix
        disp =ConfusionMatrixDisplay.from_estimator(clf, test_dataset, test_label, xticks_rotation="vertical" ,cmap="viridis")
        fig_rf = disp.figure_
        fig_rf.set_figwidth(20)
        fig_rf.set_figheight(20)
        fig_rf.suptitle('Plot of confusion matrix')
        plt.tight_layout()
        plt.savefig(f'paper/figures/4c_{run_prefix}_rf_sklearn_confusion_matrix.jpg')
        plt.show()

        # save rf model

        dump(clf, 'output/_4c_test_randomforest_trained.joblib')
        clf = load('output/_4c_test_randomforest_trained.joblib')
        print("done in %0.3fs" % (time() - t0))