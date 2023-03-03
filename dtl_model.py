import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy
import torchvision.models as models
import torch
from pytorch_lightning import seed_everything

torch.manual_seed(42)
seed = seed_everything(42, workers=True)

class DtlModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, transfer=False, num_train_layers=3, arch="resnet18"):
        super().__init__()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        self.num_train_layers = num_train_layers
        self.arch = arch
        if arch == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=transfer)
        if arch == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=transfer)
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