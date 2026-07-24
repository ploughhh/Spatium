from model import spaProFormer
from fine_tune import spaProFormerFinetune
from dataloader import MerlinFewShotDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pandas as pd
import yaml
import os
import torch
import zarr
import numpy as np

if __name__ == '__main__':
    config = {
        'pretrained_path': "/data/twang15/SPpretrain/test/checkpoints_truedata/final.ckpt", 
        'retake_training': False,
        'dim_feedforward': 512,
        'nheads': 16,
        'masking_p': 0.15,
        'nlayers': 12,
        'dropout': 0.2,
        'dim_model': 256,
        'batch_first': True,
        'n_tokens': 278, 
        'batch_size': 256,
        'context_length': 239,
        'lr': 5e-5,
        'weight_decay': 0.1,
        'warmup': 50000, 
        'max_epochs': 50, 
        # 'task': 'cell_type_prediction',
        'task': 'Prototype_classification',
        # 'task': 'neighborhood_identify',
        'finetune_mode': 'full',
        'autoregressive': False,
        'pool': None,
        'supervised_task': False,
        'learnable_pe': True,
        'organ': "everything",
        'specie': True,
        'assay': True,
        'modality': True,
        }


    save_ckpt_dir = '/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/ckpt'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    with open(os.path.join(save_ckpt_dir, "train_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    pl.seed_everything(42)

    # parquet_files = [
    #     '/data/twang15/SPpretrain/fine_tune/CyCIF_CRC/train.parquet',
    #     # '/data/twang15/SPpretrain/fine_tune/CyCIF_CRC/val.parquet'
    # ]
    # zarr_path = '/data/twang15/SPpretrain/fine_tune/IMC_COAD/zarr'
    # zarr_path = [
    #     '/data/twang15/SPpretrain/fine_tune/IMC_COAD/zarr1',
    #     '/data/twang15/SPpretrain/fine_tune/IMC_COAD/zarr2'
    # ]
    # zarr_path = '/data/twang15/SPpretrain/fine_tune/CODEX_tonsil/input'
    zarr_path = '/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/input'
    num_cells = zarr.open(zarr_path, mode='r').shape[0]
    num_cell_types = len(np.unique(zarr.open(zarr_path, mode='r')[:, config['context_length']:]))
    # zarr_path = [
    #     '/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/input',
    #     '/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/val_input'
    # ]
    # DataModule: has_label=True for fine-tune parquet that contains labels
    # num_cell_types = 25  # <-- adjust to correct number (or compute from mapping)

    dm = MerlinFewShotDataModule(
        support_path='/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/fewshot/slice_0_support',
        query_path='/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/fewshot/slice_0_query',
        batch_size=config['batch_size'],
        task=config['task'],
        # split_by_file=False,
        # context_length=config['context_length'],
        # num_types=num_cell_types,
        # k=30
    )
    dm.setup()

    # -----------------------
    # Load pre-trained SPA model
    # -----------------------
    pretrained_ckpt = "/data/twang15/SPpretrain/test/checkpoints_truedata/final.ckpt"

    # First try to load directly (preferred). If state dict mismatch, fallback to manual load with strict=False.
    # try:
    pretrained_model = spaProFormer.load_from_checkpoint(config['pretrained_path'])
    # -----------------------
    # Create fine-tune model using the loaded encoder/embeddings
    # -----------------------
    # YOU MUST set num_cell_types to the actual number of classes in your dataset
    fine_tune_model = spaProFormerFinetune(
        pretrained_model=pretrained_model,
        num_cell_types=num_cell_types,
        drop_out=config['dropout'],
        task=config['task'],
        lr=1e-5,
        finetune_mode=config['finetune_mode'],
        num_cells=num_cells
    )

    # -----------------------
    # Logger & Callbacks
    # -----------------------
    wandb_logger = WandbLogger(
        name=f"SProFineTune_sample_CODEX_Intestine",
        project="SProFineTune_sample_CODEX_Intestine",
        entity="tg05080930-the-university-of-texas",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_ckpt_dir,
        filename="scProFormerFineTune-{epoch:02d}-{val_accs:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # -----------------------
    # Trainer
    # -----------------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1, 
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=5,
        precision="16-mixed",
    )

    # Fit the fine-tune model (was incorrectly calling trainer.fit(model, ...) before)
    trainer.fit(fine_tune_model, datamodule=dm)
