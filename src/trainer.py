from model import spaProFormer
from dataloader import MerlinDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pandas as pd
import yaml
import os, glob
import torch

if __name__ == '__main__':
    config = {
        'pretrained_path': None, 
        'retake_training': False,
        'dim_feedforward': 512,
        'nheads': 16,
        'masking_p': 0.15,
        'nlayers': 12,
        'dropout': 0.0,
        'dim_model': 256,
        'batch_first': True,
        'n_tokens': 278, 
        'batch_size': 32,
        'context_length': 239,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'warmup': 500000, 
        'max_epochs': 1, 
        'autoregressive': False,
        'pool': None,
        'supervised_task': False,
        'learnable_pe': True,
        'organ': "everything",
        'specie': True,
        'assay': True,
        'modality': True,
        }

    os.makedirs('/data/twang15/SPpretrain/test/whole_data_pretrain', exist_ok=True)

    with open("/data/twang15/SPpretrain/test/whole_data_pretrain/train_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    pl.seed_everything(42)
    torch.backends.cudnn.benchmark = True

    zarr_path = '/data/twang15/SPpretrain/pretrain_data.zarr'

    dm = MerlinDataModule(
        zarr_path=zarr_path,
        batch_size=config['batch_size'],
        has_label=False
    )
    dm.setup()


    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = spaProFormer(
        dim_model=config['dim_model'],
        nheads=config['nheads'],
        dim_feedforward=config['dim_feedforward'],
        nlayers=config['nlayers'],
        learnable_pe=config['learnable_pe'],
        dropout=config['dropout'],
        autoregressive=config['autoregressive'],
        context_length=config['context_length'],
        batch_first=config['batch_first'],
        masking_p=config['masking_p'],
        n_tokens=config['n_tokens'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup=config['warmup'],
        batch_size=config['batch_size'],
        max_epochs=config['max_epochs'],

    )
    wandb_logger = WandbLogger(
        name=f"Pretrain_SProPretrain_256-512-8",
        project="Pretrain_SProPretrain_256-512-8",
        entity="tg05080930-the-university-of-texas",
    )

    checkpoint_callback = ModelCheckpoint(
        # monitor="val_loss",
        dirpath="/data/twang15/SPpretrain/test/checkpoints_truedata/",
        filename="scProFormer-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        # mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        strategy="ddp_find_unused_parameters_true",
        # strategy='ddp',
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50,
        precision="16-mixed",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint("/data/twang15/SPpretrain/test/checkpoints_truedata/final.ckpt")