import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy
import torchvision.models as models
import torch
from pytorch_lightning import seed_everything
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Dict, Any, Sequence, Union, List, Optional
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
torch.manual_seed(42)
seed = seed_everything(42, workers=True)

class DtlModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, label_encoder, learning_rate=2e-4, transfer=False, num_train_layers=3, arch="resnet18", wandb_name=None):
        """
        Model for Deep Learning

        Args:
            input_shape:
            num_classes:
            learning_rate:
            transfer:
            num_train_layers:
            arch:
        """
        super().__init__()
        if wandb_name:
            self.wandb_name = wandb_name
        else:
            self.wandb_name = "experiment"
        self.label_encoder = label_encoder
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        self.num_train_layers = num_train_layers
        self.arch = arch
        if arch == "resnet18":
            # ToDo fix because we use an classification layer add the forward step
            resnet = models.resnet18(pretrained=transfer)
            resnet.fc = nn.Linear(512, self.num_classes)
            print(resnet)
            #model = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_extractor = resnet
        elif arch == "resnet_dino":
            c_path = "saved_models/epoch=299-step=34200.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            student_backbone_state_dict = OrderedDict()
            stlen = len("teacher_backbone.")
            for key, value in resnet_weights.items():
                if key.startswith('teacher_backbone'):
                    student_backbone_state_dict[key[stlen:]] = value
            resnet = models.resnet18(pretrained=False)
            model = nn.Sequential(*list(resnet.children())[:-1])
            # load the extracted teacher backbone weights into the new model
            model.load_state_dict(student_backbone_state_dict)
            model = nn.Sequential(model, nn.Flatten(), nn.Linear(resnet.fc.in_features, self.num_classes))
            self.feature_extractor=model
            #torch.save(resnet_weights, 'resnet_backbone_weights.pth')
            #model.load_state_dict(checkpoint["state_dict"])
        elif arch == "resnet_dino450":
            c_path = "saved_models/epoch=449-step=23850.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            student_backbone_state_dict = OrderedDict()
            stlen = len("teacher_backbone.")
            for key, value in resnet_weights.items():
                if key.startswith('teacher_backbone'):
                    student_backbone_state_dict[key[stlen:]] = value
            resnet = models.resnet18(pretrained=False)
            model = nn.Sequential(*list(resnet.children())[:-1])
            # load the extracted teacher backbone weights into the new model
            model.load_state_dict(student_backbone_state_dict)
            model = nn.Sequential(model, nn.Flatten(), nn.Linear(resnet.fc.in_features, self.num_classes))
            self.feature_extractor=model
        elif arch == "resnet50_dino":
            c_path = "saved_models/epoch=299-step=31800.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            student_backbone_state_dict = OrderedDict()
            stlen = len("teacher_backbone.")
            for key, value in resnet_weights.items():
                if key.startswith('teacher_backbone'):
                    student_backbone_state_dict[key[stlen:]] = value
            resnet = models.resnet50(pretrained=False)
            model = nn.Sequential(*list(resnet.children())[:-1])
            # load the extracted teacher backbone weights into the new model
            model.load_state_dict(student_backbone_state_dict)
            model = nn.Sequential(model, nn.Flatten(), nn.Linear(resnet.fc.in_features, self.num_classes))
            self.feature_extractor = model
        elif arch == "dtl_resnet18_classifier2":#old
            c_path = "loki/ghz801ed/checkpoints/epoch=2-step=456.ckpt"
            #c_path = "loki/yhljj2fz/checkpoints/epoch=3-step=608.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            model = models.resnet18(pretrained=False)
            print(resnet_weights.keys())
            print(model.state_dict().keys())
            #model.fc = nn.Linear(512, self.num_classes)
            model = nn.Sequential(
                model,
                nn.Linear(1000, self.num_classes)
            )
            real_keys = model.state_dict().keys()
            new_values = resnet_weights.values()
            print(len(real_keys))
            print(len(new_values))
            my_ordered_dict = OrderedDict(zip(real_keys, new_values))
            model.load_state_dict(my_ordered_dict)
            print(model)
            self.feature_extractor=model
        elif arch == "dtl_resnet18_classifier":
            #c_path = "loki/ghz801ed/checkpoints/epoch=2-step=456.ckpt"
            c_path = "loki/yhljj2fz/checkpoints/epoch=3-step=608.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            model = models.resnet18(pretrained=False)
            real_keys = model.state_dict().keys()
            new_values = resnet_weights.values()
            my_ordered_dict = OrderedDict(zip(real_keys, new_values))
            print(len(real_keys))
            print(len(new_values))
            model.fc = nn.Linear(512, self.num_classes)
            model.load_state_dict(my_ordered_dict)
            print(model)
            self.feature_extractor=model
        # forward ohne classifier
        elif arch == "dino_resnet18_classifier":
            c_path = "loki/kbnoao6c/checkpoints/epoch=4-step=760.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            model = models.resnet18(pretrained=False)
            real_keys = model.state_dict().keys()
            new_values = resnet_weights.values()
            my_ordered_dict = OrderedDict(zip(real_keys, new_values))
            print(len(real_keys))
            print(len(new_values))
            model.fc = nn.Linear(512, self.num_classes)
            model.load_state_dict(my_ordered_dict)
            print(model)
            self.feature_extractor=model
        elif arch == "dino_resnet18_classifier_10":
            #num3, bzw 2 und classifier
            c_path = "loki/227395ze/checkpoints/epoch=6-step=1064.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            model = models.resnet18(pretrained=False)
            real_keys = model.state_dict().keys()
            new_values = resnet_weights.values()
            my_ordered_dict = OrderedDict(zip(real_keys, new_values))
            print(len(real_keys))
            print(len(new_values))
            model.fc = nn.Linear(512, self.num_classes)
            model.load_state_dict(my_ordered_dict)
            print(model)
            self.feature_extractor=model
        elif arch == "dtl_resnet18_classifier_10":
            c_path = "loki/m8zhox5b/checkpoints/epoch=3-step=608.ckpt"
            checkpoint = torch.load(c_path, map_location=torch.device('mps'))
            resnet_weights = checkpoint["state_dict"]
            model = models.resnet18(pretrained=False)
            real_keys = model.state_dict().keys()
            new_values = resnet_weights.values()
            my_ordered_dict = OrderedDict(zip(real_keys, new_values))
            print(len(real_keys))
            print(len(new_values))
            model.fc = nn.Linear(512, self.num_classes)
            model.load_state_dict(my_ordered_dict)
            print(model)
            self.feature_extractor=model
        self.save_hyperparameters()
        if transfer:
            max_Children = int(len([child for child in self.feature_extractor.children()]))
            ct = max_Children
            for child in self.feature_extractor.children():
                ct -= 1
                if ct < self.num_train_layers:
                    for param in child.parameters():
                        param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()


    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        """

        Args:
            shape:

        Returns:

        """
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = self.feature_extractor(x)
        return x

    def forward(self, x):
       """

       Args:
           x:

       Returns:

       """
       x = self._forward_features(x)
       #x = x.view(x.size(0), -1)
       #x = self.classifier(x)
       return x

    def training_step(self, batch:torch.Tensor):
        """
        Overwrite the lightning training step method.

        Args:
            batch: tensor
        Returns:
            dict with loss, outputs and gt
        """
        return self._shared_eval_step(batch, "train")

    def validation_step(self, batch:torch.Tensor, batch_idx:int):
        """
        Overwrite the lightning validation step method.

        Args:
            batch:
            batch_idx:
        Returns:
            dict with loss, outputs and gt
        """
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch:torch.Tensor, batch_idx:int):
        """
        Overwrite the lightning test step method.

        Args:
            batch:
            batch_idx:
        Returns:
            dict with loss, outputs and gt
        """
        return self._shared_eval_step(batch, "test")

    def _shared_eval_step(self, batch:torch.Tensor, mode:str):
        """
        Extract the img and the gt from the batch

        Args:
            batch: tensor
            mode: train, test or val
        Returns:
            dict with loss, outputs and gt
        """
        batch_x, gt = batch[0], batch[1]
        out = self.forward(batch_x)#ToDo fix
        loss = self.criterion(out, gt)
        self.log(f"{mode}/loss", loss, prog_bar=True)
        self.log(f"{mode}/acc", self.accuracy(out, gt), prog_bar=True)
        return {"loss": loss, "outputs": out, "gt": gt}

    def _epoch_end(self, outputs:torch.Tensor, mode:str):
        """
        Calculate loss and acc at end of epoch

        Args:
            outputs:
            mode:
        Returns:
            None
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        output = torch.cat([x["outputs"] for x in outputs], dim=0)
        gts = torch.cat([x["gt"] for x in outputs], dim=0)
        self.log(f"{mode}/loss", loss)
        self.log(f"{mode}/acc", self.accuracy(output, gts))
        if mode == "test":
            self._sklearn_metrics(outputs, gts, mode)

    def _sklearn_metrics(self, outputs:torch.Tensor, gts:torch.Tensor, mode:str):
        """
        Calculate the sklearn classification report, the confusion matrix and the scores for the test data

        Args:
            outputs:
            gts:
            mode:
        Returns:
            None
        """
        softmax = nn.Softmax(dim=1)
        output = torch.cat([x["outputs"] for x in outputs], dim=0)
        preds = torch.cat([softmax(x["outputs"]).argmax(dim=1) for x in outputs], dim=0)
        confis = softmax(output).detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()
        y_pred = self.label_encoder.inverse_transform(y_pred)
        y_true = gts.detach().cpu().numpy()
        y_true = self.label_encoder.inverse_transform(y_true)
        report = classification_report(y_true, y_pred, output_dict=True)
        report_confusion_matrix = confusion_matrix(y_true, y_pred, labels=list(self.label_encoder.classes_))
        model_dir = "paper/runs/"
        df_cm = pd.DataFrame(report_confusion_matrix)
        df_cr = pd.DataFrame(report).transpose()
        disp = ConfusionMatrixDisplay(confusion_matrix=report_confusion_matrix, display_labels=self.label_encoder.classes_).plot(cmap=plt.cm.Blues)
        fig_rf = disp.figure_
        fig_rf.set_figwidth(20)
        fig_rf.set_figheight(20)
        #fig_rf.suptitle('Plot of confusion matrix')
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(f'{model_dir}/{self.wandb_name}_{mode}_confusion_matrix.jpg')
        plt.show()
        df_cm.to_csv(f"{model_dir}/{self.wandb_name}_{mode}_confusion_matrix.csv", sep=";")
        df_cr.to_csv(f"{model_dir}/{self.wandb_name}_{mode}_classification_report.csv", sep=";")
        df_test = pd.DataFrame()
        df_test['id'] = [i for i in range(len(output))]
        df_test['pred'] = y_pred
        df_test['confidence'] = confis.max(axis=1)
        df_test['label'] = y_true
        df_test.to_csv(f"{model_dir}/{self.wandb_name}_{mode}_labels_predictions.csv", sep=";")

    def test_epoch_end(self, outputs:torch.Tensor):
        """
        Calculate the metrics at the end of epoch for testing step

        Args:
            outputs:
        Returns:
            None
        """
        self._epoch_end(outputs, "test")

    def validation_epoch_end(self, outputs:torch.Tensor):
        """
        Calculate the metrics at the end of epoch for validation step

        Args:
            outputs:
        Returns:
            None
        """
        self._epoch_end(outputs, "val")

    def predict_step(self, batch:torch.Tensor, batch_idx:int, dataloader_idx:int=0)->Dict[str, torch.Tensor]:
        """
        Calculate model prediction

        Args:
            outputs:
        Returns:
            None
        """
        batch_x, gt, file_name = batch[0], batch[1], batch[2]
        out = self.forward(batch_x)#
        softmax = nn.Softmax(dim=1)
        preds = softmax(out).argmax(dim=1).detach().cpu().numpy()
        preds = self.label_encoder.inverse_transform(preds)
        confis = np.max(softmax(out).detach().cpu().numpy(), axis=1)
        return {"file_names": file_name, "preds": preds, "confis":confis}

    def lr_scheduler_step(
        self,
        scheduler:Any,
        optimizer_idx: int,
        metric: Optional[Any],
    ) -> None:
        """
        LR Sheduler Step
        Args:
            scheduler:
            optimizer_idx:
            metric:
        Returns:
            None
        """
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self)->Any:
        """
        Configure the adam optimizer
        Returns:
            optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=100)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def configure_callbacks(self) -> Union[Sequence[pl.callbacks.Callback], pl.callbacks.Callback]:
        """
        Configure the call back e.g. Early stopping or Model Checkpointing
        Returns:
        """
        early_stop = EarlyStopping(monitor ="val/acc", patience=3,  mode = "max")
        checkpoint = ModelCheckpoint(monitor ="val/acc", mode = "max")
        return [early_stop, checkpoint]
