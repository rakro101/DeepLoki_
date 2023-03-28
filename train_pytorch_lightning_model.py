import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from loki_datasets import LokiDataModule, LokiTrainValDataset
from dtl_model import DtlModel

from pytorch_lightning.loggers import WandbLogger

import wandb
wandb.login()

torch.manual_seed(42)
seed = seed_everything(42, workers=True)


if __name__ == '__main__':
    dm = LokiDataModule(batch_size=1024)#1024
    lrvd = LokiTrainValDataset()
    num_classes = lrvd.n_classes
    label_encoder = lrvd.label_encoder
    logger = WandbLogger(project="loki")
    print(wandb.run.name)
    model = DtlModel(input_shape=(3,300,300), label_encoder=label_encoder, num_classes=num_classes, arch="effnet", transfer=True, num_train_layers=3, wandb_name=wandb.run.name, learning_rate=5.7543993733715664e-05)
    bs_fit = False
    lr_fit = False
    if bs_fit:
        trainer=pl.Trainer(logger=logger,max_epochs=1, accelerator="mps", devices="auto", deterministic=True, auto_scale_batch_size=True)
        trainer.tune(model, dm)
    elif lr_fit:
        trainer=pl.Trainer(logger=logger,max_epochs=1, accelerator="mps", devices="auto", deterministic=True, auto_lr_find=True)
        trainer.tune(model,dm)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=10, accelerator="mps", devices="auto")
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
