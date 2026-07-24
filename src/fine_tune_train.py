from model import spaProFormer
from fine_tune import spaProFormerFinetune
from dataloader import MerlinDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pandas as pd
import yaml
import os
import torch
import zarr
import numpy as np
from callback import SaveEmbeddingCallback
os.environ["WANDB_MODE"] = "offline"

if __name__ == '__main__':
    config = {
        'pretrained_path': "/data_d/WTG/SPpretrain/final.ckpt", 
        'retake_training': False,
        'dim_feedforward': 512,
        'nheads': 16,
        'masking_p': 0.15,
        'nlayers': 12,
        'dropout': 0.2,
        'dim_model': 256,
        'batch_first': True,
        'n_tokens': 278, 
        'batch_size': 128,
        # 'batch_size': 64,
        'context_length': 239,
        'lr': 5e-5,
        'weight_decay': 0.1,
        'warmup': 50000, 
        'max_epochs': 20, 
        # 'task': 'cell_type_prediction',
        # 'task': 'Prototype_classification',
        # 'task': 'neighborhood_identify',
        'task': 'panel_expansion_continuous_new',
        # 'task': 'image_integration',
        # 'task': 'reconstruction',
        # 'task': 'label_transfer',
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


    save_ckpt_dir = '/data_d/WTG/SPpretrain/fine_tune/protein_imputation/CODEX_HuBMAP/thymus/ckpt'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    with open(os.path.join(save_ckpt_dir, "train_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    pl.seed_everything(42)


    # zarr_path = [
    #     '/data/twang15/SPpretrain/fine_tune/IMC_COAD/zarr1',
    #     '/data/twang15/SPpretrain/fine_tune/IMC_COAD/zarr2'
    # ]
    # zarr_path = '/data/twang15/SPpretrain/fine_tune/CODEX_tonsil/input'
    # zarr_path = '/data_d/WTG/SPpretrain/fine_tune/protein_imputation/CODEX_HCC/TMA4_4_reg025_new'
    zarr_path = [
        '/data_d/WTG/SPpretrain/fine_tune/protein_imputation/CODEX_HuBMAP/thymus/train',
        '/data_d/WTG/SPpretrain/fine_tune/protein_imputation/CODEX_HuBMAP/thymus/val'
    ]

    # zarr_path = [
    #     '/data/twang15/SPpretrain/fine_tune/IMC_COAD/zarr1',
    #     '/data/twang15/SPpretrain/fine_tune/IMC_COAD/zarr2'
    # ]
    # zarr_path = [
    #     '/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/input',
    #     '/data/twang15/SPpretrain/fine_tune/Nature_CODEX_intestine/val_input'
    # ]
    # DataModule: has_label=True for fine-tune parquet that contains labels
    if isinstance(zarr_path, list):
        z = zarr.open(zarr_path[0], mode='r')
    else:
        z = zarr.open(zarr_path, mode='r')
    if config['task'] == 'neighborhood_identify':
        z = zarr.open(zarr_path, mode='r')
        if 'neighbor_ratio' in z and z.get('neighbor_ratio') is not None:
            num_cell_types = np.array(z.get('neighbor_ratio')).shape[1]
    elif config['task'] == 'label_transfer':
        z = zarr.open(zarr_path[0], mode='r')
        labels = np.array(z.get('label'))
        num_cell_types = labels.max() + 1
    elif 'label' in z and z.get('label') is not None:
        z = zarr.open(zarr_path, mode='r')
        labels = np.array(z.get('label'))
        num_cell_types = labels.max() + 1
    else:
        num_cell_types = None

    dm = MerlinDataModule(
        zarr_path=zarr_path,
        batch_size=config['batch_size'],
        task=config['task'],
        split_by_file=True,
        context_length=config['context_length'],
    )
    dm.setup()


    pretrained_model = spaProFormer.load_from_checkpoint(config['pretrained_path'])

    if config['task'] == 'panel_expansion_continuous':
        continuous_dim = zarr.open(zarr_path, mode='r').get('continuous').shape[1]
    elif config['task'] == 'panel_expansion_continuous_new':
        continuous_dim = zarr.open(zarr_path[0], mode='r').get('continuous').shape[1]
    else:
        continuous_dim = None
    fine_tune_model = spaProFormerFinetune(
        pretrained_model=pretrained_model,
        num_cell_types=int(num_cell_types) if num_cell_types is not None else None,
        drop_out=config['dropout'],
        task=config['task'],
        lr=1e-5,
        finetune_mode=config['finetune_mode'],
        continuous_dim=continuous_dim,
        graph_pe_dim=8
    )

    wandb_logger = WandbLogger(
        name=f"SProFineTune_sample_CODEX_Kidney_imputation",
        project="SProFineTune_sample_CODEX_Kidney_imputation",
        entity="tg05080930-the-university-of-texas",
    )

    checkpoint_callback = ModelCheckpoint(
        # monitor="val_loss",
        dirpath=save_ckpt_dir,
        filename="scProFormerFineTune-{epoch:02d}-{val_accs:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=5,
        precision="16-mixed",
        use_distributed_sampler=False if config['task'] == 'pancancer_engine' else True,
        num_sanity_val_steps=0,

    )


    trainer.fit(fine_tune_model, datamodule=dm)
