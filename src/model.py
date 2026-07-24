import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List, Dict, Optional, Union
from torch import optim
import numpy as np
import math
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import _LRScheduler
from _utils import complete_masking
from constants import PROTEIN_TOKEN_BASE

class spaProFormer(pl.LightningModule):
    def __init__(self,
                 dim_model: int,
                 nheads: int,
                 dim_feedforward: int,
                 nlayers: int,
                 batch_first: bool,
                 masking_p: float,
                 n_tokens: int,
                 context_length: int,
                 lr: float,
                 weight_decay: float,
                 warmup: int,
                 batch_size: int,
                 max_epochs: int,
                 dropout: float,
                 cls_classes: int,
                 supervised_task: Optional[str] = None,
                 learnable_pe: bool = True,
                 specie: bool = False,
                 assay: bool = False,
                 modality: bool = False,
                 contrastive: bool = False,
                 autoregressive: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.dim_model = dim_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.nlayers = nlayers
        self.batch_first = batch_first
        self.masking_p = masking_p
        self.n_tokens = n_tokens + 1
        self.context_length = context_length
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.cls_classes = cls_classes
        self.supervised_task = supervised_task
        self.learnable_pe = learnable_pe
        self.specie = specie
        self.assay = assay
        self.modality = modality
        self.contrastive = contrastive
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.autoregressive = autoregressive
        self._transformers()
        self._embeddings()
        self._model_heads()
        self.loss = nn.CrossEntropyLoss()
        self.initialize_weights()


    def _transformers(self):
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=self.nheads,
            dim_feedforward=self.dim_feedforward,
            batch_first=self.batch_first,
            dropout=self.dropout_rate,
            layer_norm_eps=1e-12
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layers,
            num_layers=self.nlayers,
            enable_nested_tensor=False
        )
    def _embeddings(self):
        self.embeddings = nn.Embedding(
            num_embeddings=self.n_tokens,
            embedding_dim=self.dim_model,
            padding_idx=0
        )
        if self.learnable_pe:
            self.positional_embedding = nn.Embedding(
                num_embeddings=self.context_length,
                embedding_dim=self.dim_model
            )
            self.pos = torch.arange(0, self.context_length, dtype=torch.long)
        else:
            self.positional_embedding = PositionalEncoding(
                d_model=self.dim_model,
                max_seq_len=self.context_length
            )
    def _model_heads(self):
        self.classif_head = nn.Linear(self.dim_model, self.n_tokens, bias=False)
        self.classif_head.bias = nn.Parameter(torch.zeros(self.n_tokens))
        self.pool_head = nn.Linear(self.dim_model, self.dim_model)
        self.activation = nn.Tanh()
    
    def forward(self, x, attention_mask):
        token_embeddings = self.embeddings(x)
        if self.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embeddings.device))
            embeddings = self.dropout(token_embeddings + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embeddings)

        transformer_output = self.encoder(
            embeddings,
            src_key_padding_mask=attention_mask
        )

        prediction = self.classif_head(transformer_output)

        return {
            'mlm_prediction': prediction,
            'transformer_output': transformer_output
        }

    def training_step(self, batch):
        with torch.no_grad():
            masked_indices, mask, attention_mask, real_indices = complete_masking(batch, n_tokens=self.n_tokens)

        predictions = self.forward(masked_indices, attention_mask)
        mlm_predictions = predictions['mlm_prediction']

        real_indices = self._prepare_target_indices(mask, real_indices)
        loss = self._compute_loss(mlm_predictions, real_indices)

        mlm_acc = self._compute_mask_accuracy(mlm_predictions, real_indices)

        self.log('train_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        self.log('train_mlm_acc', mlm_acc, sync_dist=True, prog_bar=True, reduce_fx='mean')
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            masked_indices, mask, attention_mask, real_indices = complete_masking(batch, n_tokens=self.n_tokens)
        
        predictions = self.forward(masked_indices, attention_mask)
        mlm_predictions = predictions['mlm_prediction']

        real_indices = self._prepare_target_indices(mask, real_indices)
        loss = self._compute_loss(mlm_predictions, real_indices)

        mlm_acc = self._compute_mask_accuracy(mlm_predictions, real_indices)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        self.log('val_mlm_acc', mlm_acc, sync_dist=True, prog_bar=True, reduce_fx='mean')
        return loss

    def _prepare_target_indices(self, mask: torch.Tensor, real_indices: torch.Tensor) -> torch.Tensor:
        """Prepare target indices for loss computation. Only masked indices are taken into account."""
        return torch.where(
            mask == 1,
            real_indices,
            torch.tensor(-100, dtype=torch.long, device=real_indices.device)
        ).type(torch.int64)

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for training/validation."""
        predictions = predictions.view(-1, self.n_tokens)
        targets = targets.view(-1)

        if self.masking_p == 0.0:
            return torch.tensor(0.0, device=predictions.device)
        return self.loss(predictions, targets)
    
    def _compute_mask_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        predictions: [batch, seq_len, n_tokens]
        targets: [batch, seq_len]
        """
        with torch.no_grad():
            preds = predictions.argmax(dim=-1)
            mask = targets != -100
            if mask.sum() == 0:
                return torch.tensor(0.0, device=predictions.device)
            correct = (preds == targets) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        return accuracy
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.warmup, max(int(0.01 * total_steps), 1000))

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions."""
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)

        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.encoding[:, :x.size(1)]


class CosineWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine decay (per step), Lightning compatible.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            # linear warmup: 0 -> base_lr
            factor = step / max(1, self.warmup_steps)
        else:
            # cosine decay: base_lr -> 0
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = max(0.0, factor)  
        return [base_lr * factor for base_lr in self.base_lrs]