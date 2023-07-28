import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from loki_datasets import LokiDataModule, LokiTrainValDataset
from dtl_model import DtlModel
import time
from pytorch_lightning.loggers import WandbLogger
import glob
import wandb
wandb.login()

torch.manual_seed(42)
seed = seed_everything(42, workers=True)


if __name__ == '__main__':
    dm = LokiDataModule(batch_size=1536)#1024
    lrvd = LokiTrainValDataset()
    num_classes = lrvd.n_classes
    label_encoder = lrvd.label_encoder
    logger = WandbLogger(project="loki")
    print(wandb.run.name)
    print(wandb.run.id)
    model = DtlModel(input_shape=(3,300,300), label_encoder=label_encoder, num_classes=num_classes, arch="resnet_dino450", transfer=True, num_train_layers=1, wandb_name=wandb.run.name, learning_rate=0.0001)#5.7543993733715664e-05)
    bs_fit = False
    lr_fit = False
    if bs_fit:
        trainer=pl.Trainer(logger=logger,max_epochs=1, accelerator="mps", devices="auto", deterministic=True, auto_scale_batch_size=True)
        trainer.tune(model, dm)
    elif lr_fit:
        trainer=pl.Trainer(logger=logger,max_epochs=1, accelerator="mps", devices="auto", deterministic=True, auto_lr_find=True)
        trainer.tune(model,dm)
    else:
        trainer = pl.Trainer(precision=16, logger=logger, max_epochs=20, accelerator="mps", devices="auto", deterministic=True)
        trainer.fit(model, dm)
        folder_path = f'loki/{wandb.run.id}/checkpoints/'
        file_pattern = folder_path + '*.ckpt'
        file_list = glob.glob(file_pattern)
        print(file_list[0])
        trainer.validate(model, dm, ckpt_path=file_list[0])
        trainer.test(model, dm, ckpt_path=file_list[0])
