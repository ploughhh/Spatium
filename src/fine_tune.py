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
import torch.nn.functional as F
from _utils import complete_masking
from model import spaProFormer
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional import pearson_corrcoef
from scipy.stats import spearmanr, pearsonr
from torch.autograd import Function
from sklearn.metrics import accuracy_score, f1_score

def inter_class_ot_loss(prototypes, metric="cosine", eps=0.1):
    """
    prototypes: [num_classes, dim_model]
    metric: "cosine" or "euclidean"
    """
    num_classes = prototypes.shape[0]

    proto_norm = F.normalize(prototypes, dim=-1) if metric=="cosine" else prototypes

    if metric == "cosine":
        dist_matrix = 1 - proto_norm @ proto_norm.T  # cosine distance
    else:
        diff = proto_norm.unsqueeze(1) - proto_norm.unsqueeze(0)  # [C, C, D]
        dist_matrix = (diff ** 2).sum(-1)  # squared euclidean

    mask = 1 - torch.eye(num_classes, device=prototypes.device)
    dist_matrix = dist_matrix * mask

    loss = F.relu(eps - dist_matrix).mean()
    return loss

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    Args:
        gamma: focusing parameter, default 2.0
        alpha: class weighting factor, can be float or list/array for each class
        reduction: 'mean' or 'sum'
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B, C] raw output
        targets: [B] long, 0..C-1
        """
        B, C = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)                # [B, C]

        targets_onehot = F.one_hot(targets, num_classes=C).float()  # [B, C]

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if alpha.shape[0] == 1:
                alpha = alpha.repeat(C)
            alpha_factor = alpha.unsqueeze(0)  # [1, C]
        else:
            alpha_factor = 1.0

        focal_weight = (1 - probs) ** self.gamma
        loss = -alpha_factor * focal_weight * targets_onehot * log_probs

        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=1)  # per-sample

def topk_loss(pred, target, mask, k_ratio=0.2):
    pred_masked = pred[mask]
    target_masked = target[mask]

    k = max(1, int(len(pred_masked) * k_ratio))
    pred_topk = torch.topk(pred_masked, k).values
    target_topk = torch.topk(target_masked, k).values

    return F.mse_loss(pred_topk, target_topk)


def contrastive_loss_fn(z_shared, disease_labels, temperature=0.1):
    """
    z_shared: [B, dim], pooled shared embedding
    disease_labels: [B]
    """
    z = F.normalize(z_shared, dim=-1)
    B = z.size(0)

    sim_matrix = torch.matmul(z, z.T) / temperature
    mask_diag = torch.eye(B, device=z.device, dtype=torch.bool)

    sim_matrix = sim_matrix.masked_fill(mask_diag, -1000.0)

    disease_matrix = disease_labels.view(-1,1) == disease_labels.view(1,-1)
    pos_mask = ~disease_matrix
    sim_exp = torch.exp(sim_matrix) * pos_mask.float()
    sim_sum = sim_exp.sum(dim=1, keepdim=True) + 1e-8
    log_prob = torch.log(sim_exp / sim_sum + 1e-8)
    loss = -log_prob.sum(dim=1).mean()
    return loss

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

def gaussian_kernel(x, y, sigma=1.0):
    # x: [N, D], y: [M, D]
    x = x.unsqueeze(1)  # [N,1,D]
    y = y.unsqueeze(0)  # [1,M,D]
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))

def mmd_loss(x, y, sigma=1.0):
    Kxx = gaussian_kernel(x, x, sigma).mean()
    Kyy = gaussian_kernel(y, y, sigma).mean()
    Kxy = gaussian_kernel(x, y, sigma).mean()
    return Kxx + Kyy - 2 * Kxy

def rank_loss(pred, target):
    diff_pred = pred.unsqueeze(-1) - pred.unsqueeze(-2)
    diff_true = target.unsqueeze(-1) - target.unsqueeze(-2)
    sign = torch.sign(diff_true)
    return F.relu(-sign * diff_pred).mean()

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask):
        """
        x: B×L×D
        mask: B×L (True = valid)
        """

        scores = self.score(x).squeeze(-1)  # B×L

        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=1)  # B×L

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return pooled

def neighbor_swap_topk(x, topk=10, swap_prob=0.2):
    x = x.clone()
    B, L = x.shape
    k = min(topk, L - 1)
    for b in range(B):
        i = 0
        while i < k - 1:
            if torch.rand(1, device=x.device) < swap_prob:
                tmp = x[b, i].clone()
                x[b, i] = x[b, i + 1]
                x[b, i + 1] = tmp
                i += 2
            else:
                i += 1
    return x


class spaProFormerFinetune(pl.LightningModule):
    def __init__(self,
                 pretrained_model: spaProFormer,
                 drop_out: float = 0.0,
                 task: str = None,
                 lr: float = 1e-4,
                 num_cell_types: int = None,
                 finetune_mode: str = "full",
                 tau=0.1,
                 continuous_dim: int = None,
                 graph_pe_dim: int = None,
                 label_transfer_prior: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])
        self.encoder = pretrained_model.encoder
        self.embeddings = pretrained_model.embeddings
        self.positional_embedding = pretrained_model.positional_embedding
        self.learnable_pe = pretrained_model.learnable_pe
        self.pos = pretrained_model.pos if pretrained_model.learnable_pe else None
        self.context_length = pretrained_model.context_length
        self.dropout = nn.Dropout(drop_out)
        self.dim_model = pretrained_model.dim_model
        self.layernorm_input = nn.LayerNorm(self.dim_model)
        self.layernorm_output = nn.LayerNorm(self.dim_model)
        self.task = task
        print(self.task)
        self.n_tokens = pretrained_model.n_tokens
        self.continuous_dim = continuous_dim
        if self.task == "cell_type_prediction":
            assert num_cell_types is not None, "You must input num_cell_types"
            self.pooler = nn.Linear(self.dim_model, self.dim_model)
            self.activation = nn.Tanh()
            self.classifier = nn.Linear(self.dim_model, num_cell_types)
            self.loss_fn = nn.CrossEntropyLoss()
            self.train_f1 = MulticlassF1Score(num_classes=num_cell_types, average='macro')
            self.val_f1 = MulticlassF1Score(num_classes=num_cell_types, average='macro')
            self.train_acc = Accuracy(task="multiclass", num_classes=num_cell_types)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_cell_types)
        elif self.task == 'Prototype_classification':
            assert num_cell_types is not None, "You must input num_cell_types"
            self.tau = tau
            self.prototypes = nn.Parameter(torch.randn(num_cell_types, self.dim_model))
            nn.init.kaiming_normal_(self.prototypes)
            # self.loss_fn = FocalLoss(gamma=2.0, alpha=alpha, reduction='mean')
            self.loss_fn = nn.CrossEntropyLoss()
            self.train_f1 = MulticlassF1Score(num_classes=num_cell_types, average='macro')
            self.val_f1 = MulticlassF1Score(num_classes=num_cell_types, average='macro')
            self.train_acc = Accuracy(task="multiclass", num_classes=num_cell_types)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_cell_types)
        elif self.task == 'neighborhood_identify':
            self.pooler = nn.Linear(self.dim_model, self.dim_model)
            self.composition_head = nn.Sequential(
                nn.Linear(self.dim_model, self.dim_model),
                nn.ReLU(),
                nn.Dropout(drop_out),
                nn.Linear(self.dim_model, num_cell_types)
            )
            # regression loss
            self.reg_loss = nn.MSELoss()
        elif self.task == 'panel_expansion':
            self.mlm_head = pretrained_model.classif_head
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.task == 'panel_expansion_continuous':
            self.decoder = nn.Linear(self.dim_model, self.continuous_dim)
            self._train_cont_preds = []
            self._train_cont_targets = []
            self._train_cont_masks = []

            self._val_cont_preds = []
            self._val_cont_targets = []
            self._val_cont_masks = []
            
        elif self.task == 'panel_expansion_continuous_new':
            self.decoder = nn.Linear(self.dim_model, self.continuous_dim)
            self.proj = nn.Sequential(
                nn.Linear(self.dim_model, self.dim_model),
                nn.ReLU(),
                nn.Linear(self.dim_model, 1)
                )

            # self.rank_embedding = nn.Embedding(240, self.dim_model)
            self._train_cont_preds = []
            self._train_cont_targets = []
            self._train_cont_masks = []

            self._val_cont_preds = []
            self._val_cont_targets = []
            self._val_cont_masks = []

            
        elif self.task == 'image_integration':
            self.graph_pe_dim = graph_pe_dim
            self.img_head = nn.Sequential(
                nn.Linear(50, 512),
                nn.ReLU(),
                nn.Linear(512, self.dim_model)
            )
            self.PE_mlp = nn.Sequential(
                nn.Linear(self.graph_pe_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.dim_model)
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.dim_model * 3, 512),  # 768 -> 512
                nn.ReLU(),
                nn.Linear(512, self.dim_model)       # 512 -> 256
            )
            # self.decoder = nn.Linear(self.dim_model, self.dim_model+50)
            self.decoder = nn.Linear(self.dim_model, self.dim_model)
            self.reg_loss = nn.MSELoss()
        elif self.task == 'reconstruction':
            self.mlm_head = pretrained_model.classif_head
            self.loss_fn = nn.CrossEntropyLoss()

        elif self.task == 'label_transfer':
            self.label_classifier = nn.Sequential(
                nn.Linear(self.dim_model, 128),
                nn.ReLU(),
                nn.Linear(128, num_cell_types)
            )

            self.proj = nn.Sequential(
                nn.Linear(self.dim_model, self.dim_model),
                nn.ReLU(),
                nn.Linear(self.dim_model, 1)
                )

            self._val_cont_preds = []
            self._val_cont_targets = []
            self._val_cont_masks = []

            self.mlm_head = pretrained_model.classif_head
            self.loss_fn = nn.CrossEntropyLoss()
            self.val_f1 = MulticlassF1Score(num_classes=num_cell_types, average='macro')
            self.val_acc = Accuracy(task="multiclass", num_classes=num_cell_types)


        self.lr = lr
        self.finetune_mode = finetune_mode.lower()
        self._apply_mode_freeze()

    def _apply_mode_freeze(self):

        mode = self.finetune_mode
        if mode == "linear_probe":
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.embeddings.parameters():
                param.requires_grad = False
            if self.learnable_pe:
                for param in self.positional_embedding.parameters():
                    param.requires_grad = False

        elif mode == "partial":

            if hasattr(self.encoder, "layers"):
                n_layers = len(self.encoder.layers)
                unfreeze_layers = 4
                for i, layer in enumerate(self.encoder.layers):
                    requires_grad = (i >= n_layers - unfreeze_layers)
                    for param in layer.parameters():
                        param.requires_grad = requires_grad

        elif mode == "full":
            pass

        else:
            raise ValueError(f"Unknown finetune_mode '{mode}'")

    def aggregate(self, logits, mask, k=10):
        score = logits.mean(-1)

        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)

        topk = torch.topk(score, k=k, dim=1).indices

        batch_idx = torch.arange(logits.size(0)).unsqueeze(-1)

        selected = logits[batch_idx, topk]

        return selected.mean(dim=1)

    def forward(self, x, attention_mask=None):
        token_embeddings = self.embeddings(x)
        if self.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embeddings.device))
            embeddings = token_embeddings + pos_embedding
        else:
            embeddings = self.positional_embedding(token_embeddings)
        embeddings = self.layernorm_input(embeddings)
        embeddings = self.dropout(embeddings)


        transformer_output = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        if self.task == "cell_type_prediction":
            pooled_output = transformer_output[:, 0]  # [CLS] pooling
            pooled_output = self.activation(self.pooler(pooled_output))
            logits = self.classifier(pooled_output)
            return logits
        elif self.task == 'Prototype_classification':
            pooled_output = transformer_output[:, 0]
            return pooled_output
        elif self.task == 'neighborhood_identify':
            pooled_output = transformer_output[:, 0]
            return pooled_output
        elif self.task == 'panel_expansion':
            pooled_output = self.mlm_head(transformer_output)
            return pooled_output
        elif self.task == 'panel_expansion_continuous':
            pooled_output = self.decoder(transformer_output)
            return pooled_output.squeeze(-1)
        elif self.task == 'panel_expansion_continuous_new':
            transformer_output = transformer_output[:, 5:self.continuous_dim+5, :]
            score = self.proj(transformer_output).squeeze(-1)
            return score
        elif self.task == 'image_integration':
            pooled_output = transformer_output[:, 0]
            return pooled_output
        elif self.task == 'reconstruction':
            pooled_output = transformer_output[:, 0]
            return pooled_output, transformer_output
        elif self.task == 'pan_cancer_engine':
            pooled_output = transformer_output[:, 0]  # CLS
            return pooled_output, transformer_output
        elif self.task == 'label_transfer':
            pooled_output = transformer_output[:, 0]  # CLS
            # transformer_output = transformer_output[:, 5:self.continuous_dim+5, :]
            # score = self.proj(transformer_output).squeeze(-1)
            # return score
            # valid_mask = ~attention_mask   # B×L

            # pooled_output = self.attention_pool(transformer_output, valid_mask)
            # # transformer_output[:, 1, :] = transformer_output[:, 1, :].detach()
            return pooled_output, transformer_output
            # return final_logits

        
    def contrastive_loss(self, embeddings, labels, tau=0.1, eps=1e-8):
        """
        Stable class-aware contrastive loss (InfoNCE)
        embeddings: [B, dim]
        labels: [B]
        """
        emb_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.matmul(emb_norm, emb_norm.T) / tau  # [B,B] cosine similarity / tau
    
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B,B] positive mask
        mask.fill_diagonal_(0)  # remove self-similarity
    
        # log-sum-exp stability
        max_sim, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_exp = torch.exp(sim_matrix - max_sim)
    
        pos_exp = sim_exp * mask.float()
        denom_exp = sim_exp.sum(dim=1) - torch.exp(torch.diagonal(sim_matrix - max_sim))  # exclude self

        has_pos = pos_exp.sum(dim=1) > 0
        valid_pos = pos_exp.sum(dim=1)[has_pos]
        valid_denom = denom_exp[has_pos]
    
        loss = -torch.log(valid_pos / (valid_denom + eps) + eps)
        if loss.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        if self.task == "cell_type_prediction":
            x = batch['x']
            labels = batch['labels']
            attention_mask = (x == 0)
            logits = self.forward(x, attention_mask)
            loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            self.train_f1.update(preds, labels)
            self.train_acc.update(preds, labels)
            self.log("train_f1", self.train_f1, on_epoch=True, on_step=False, prog_bar=True)
            self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log("train_acc", self.train_acc, on_epoch=True, on_step=False, prog_bar=True)
            return loss
        elif self.task == 'Prototype_classification':
            x = batch['x']
            labels = batch['labels']
            attention_mask = (x == 0)
            embedding = self.forward(x, attention_mask)
            emb_norm = F.normalize(embedding, dim=-1)
            proto_norm = F.normalize(self.prototypes, dim=-1)
            logits = torch.matmul(emb_norm, proto_norm.T) / self.tau
            ce_loss = self.loss_fn(logits, labels)
            contrast_loss = self.contrastive_loss(embedding, labels)
            ot_loss = inter_class_ot_loss(self.prototypes, metric="cosine", eps=0.1)
            loss = ce_loss + 0.5 * ot_loss + 0.5 * contrast_loss
            preds = torch.argmax(logits, dim=-1)
            self.train_f1.update(preds, labels)
            self.train_acc.update(preds, labels)
            self.log("train_f1", self.train_f1, on_epoch=True, on_step=False, prog_bar=True)
            self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log("train_acc", self.train_acc, on_epoch=True, on_step=False, prog_bar=True)
            return loss
        elif self.task == "neighborhood_identify":
            # cell_idx = batch["cell_idx"]            # [B]
            x = batch["x"]                      # [B, L]

            target_ratio = batch["neighbor_ratio"]  # [B, T]
            attention_mask = (x == 0)

            z_i = self.forward(x, attention_mask)

            pred_ratio = self.composition_head(z_i)

            loss = self.reg_loss(pred_ratio, target_ratio)
            self.log("train_mse", loss, on_epoch=True)
            return loss

        
        elif self.task == 'panel_expansion_continuous_new':
            x = batch['x']
            continuous = batch['continuous']
            cont_mask = batch['continuous_mask']
            bin_mat = batch['bin_mat']
            attention_mask = (x == 0)
            # cont_mask = ~cont_mask
            logits = self.forward(x, attention_mask)  # [B, L, num_proteins]
            # logits = logits.mean(dim=1)
            # loss = F.mse_loss(logits[cont_mask], continuous[cont_mask])
            # continuous_exp = continuous.unsqueeze(1).expand(-1, logits.size(1), -1)  # [B, L, P]
            # mask_exp = cont_mask.unsqueeze(1).expand(-1, logits.size(1), -1)       # [B, L, P]

            pred_masked = logits
            target_masked = continuous

            r_loss = rank_loss(pred_masked, target_masked)
            l1_loss = F.l1_loss(pred_masked, target_masked)

            K = 50
            sigma = 0.1
        
            bin_edges = torch.linspace(0, 1, K, device=logits.device)
        
            pred_norm = torch.sigmoid(logits)   # [B, P]
        
            pred_dist = torch.exp(
                - (pred_norm.unsqueeze(-1) - bin_edges) ** 2 / (2 * sigma ** 2)
            )
            pred_dist = pred_dist / (pred_dist.sum(dim=-1, keepdim=True) + 1e-8)
        
            gt_dist = F.one_hot(bin_mat, num_classes=K).float()
        

            # pred_dist_masked = pred_dist[cont_mask]   # [M, K]
            # gt_dist_masked = gt_dist[cont_mask]       # [M, K]

            pred_dist_masked = pred_dist
            gt_dist_masked = gt_dist
        
            loss_bin = F.kl_div(
                (pred_dist_masked.log() + 1e-8),
                gt_dist_masked,
                reduction='batchmean'
            )

            loss = r_loss + l1_loss + loss_bin
            # loss = r_loss + l1_loss
            



            self._train_cont_preds.append(logits.detach())
            # self._train_cont_targets.append(continuous.detach())
            # self._train_cont_masks.append(cont_mask)
            self._train_cont_targets.append(continuous.detach())
            self._train_cont_masks.append(cont_mask)

            return loss
        
        elif self.task == 'image_integration':
            x = batch['x']
            labels = batch['labels']
            HE_embedding = batch['HE_embedding']
            graph_PE = batch['graph_PE']
            attention_mask = (x == 0)
            cell_embedding = self.forward(x, attention_mask)
            image_embedding = self.img_head(HE_embedding)
            g_pe = self.PE_mlp(graph_PE)
            cell_embedding = F.normalize(cell_embedding, p=2, dim=-1)
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)
            # image_embedding = image_embedding + 0.1 * torch.randn_like(image_embedding)
            g_pe = F.normalize(g_pe, p=2, dim=-1)
            temp = 1  # attention temp
            # Query: cell_embedding, Key/Value: image_embedding
            attn_scores = torch.matmul(cell_embedding, image_embedding.T) / temp
            attn_weights = torch.softmax(attn_scores, dim=-1)
            guidance = torch.matmul(attn_weights, image_embedding)
            guidance = guidance - guidance.mean(dim=0, keepdim=True)
            g = guidance.detach().cpu().numpy()

            g_vars = np.var(g, axis=0).mean()
            alpha = 0.1
            beta = 0.0
            fused_embedding = cell_embedding + guidance * alpha + g_pe * beta
            # fused_input = torch.cat([cell_embedding, image_embedding, g_pe], dim=-1)
            # fused_embedding = self.fusion_mlp(fused_input)
            fused_embedding = F.normalize(fused_embedding, p=2, dim=-1)
            pred_features = self.decoder(fused_embedding)
            emb_dist_fused = torch.cdist(fused_embedding, fused_embedding, p=2)
            emb_dist_vit = torch.cdist(image_embedding, image_embedding, p=2)
            loss_geom = ((emb_dist_fused - emb_dist_vit)**2).mean()
            # loss = self.reg_loss(pred_features, torch.cat([cell_embedding, HE_embedding], dim=-1)) + 0.1 * loss_spatial
            loss = self.reg_loss(pred_features, cell_embedding) + 0.0 * loss_geom
            self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log('train_g_vars', g_vars, on_epoch=True, on_step=False, prog_bar=True)

            return loss
        elif self.task == 'reconstruction':
            x = batch['x']
            with torch.no_grad():
                masked_indices, mask, attention_mask, real_indices = complete_masking(
                    x,
                    n_tokens=self.n_tokens
                )

            embedding, transformer_output = self.forward(masked_indices, attention_mask)
            mlm_logits = self.mlm_head(transformer_output)
            targets = torch.where(mask == 1, real_indices, torch.tensor(-100, device=real_indices.device))
            loss = F.cross_entropy(mlm_logits.view(-1, self.n_tokens), targets.view(-1), ignore_index=-100)
            self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            return loss
        
        elif self.task == 'label_transfer':
            x = batch['x']

            labels = batch['labels']
            attention_mask = (x == 0)
            # continuous = batch['continuous']
            # # cont_mask = batch['continuous_mask']
            # bin_mat = batch['bin_mat']
            embedding, transformyer_output = self.forward(x, attention_mask)
            # embedding = self.forward(x, attention_mask)
            
            # embedding = F.normalize(embedding, dim=-1)
            dropout_prob = 0.2
            # dropout_prob = 0.0
            embedding = F.dropout(embedding, p=dropout_prob, training=self.training)
            
            # noise_std = 0.01
            # noise_std = 0.2
            noise_std = 0.2
            noise = torch.randn_like(embedding) * noise_std
            embedding = embedding + noise
            
            
            logits = self.label_classifier(embedding)
            loss = self.loss_fn(logits, labels)


            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            # self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

            return loss
        

    def validation_step(self, batch, batch_idx):
        if self.task == "cell_type_prediction":
            x = batch['x']
            labels = batch['labels']
            attention_mask = (x == 0)
            logits = self.forward(x, attention_mask)
            loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            self.val_f1.update(preds, labels)
            self.val_acc.update(preds, labels)
            self.log("val_f1", self.val_f1, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val_acc", self.val_acc, on_epoch=True, on_step=False, prog_bar=True)
            return loss
        elif self.task == 'Prototype_classification':
            x = batch['x']
            labels = batch['labels']
            attention_mask = (x == 0)
            embedding = self.forward(x, attention_mask)
            emb_norm = F.normalize(embedding, dim=-1)
            proto_norm = F.normalize(self.prototypes, dim=-1)
            logits = torch.matmul(emb_norm, proto_norm.T) / self.tau
            ce_loss = self.loss_fn(logits, labels)
            contrast_loss = self.contrastive_loss(embedding, labels)
            ot_loss = inter_class_ot_loss(self.prototypes, metric="cosine", eps=0.1)
            loss = ce_loss + 0.5 * ot_loss + 0.5 * contrast_loss
            preds = torch.argmax(logits, dim=-1)
            self.val_f1.update(preds, labels)
            self.val_acc.update(preds, labels)
            self.log("val_f1", self.val_f1, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val_acc", self.val_acc, on_epoch=True, on_step=False, prog_bar=True)
            return loss
        elif self.task == "neighborhood_identify":
            x = batch["x"]                      # [B, L]

            target_ratio = batch["neighbor_ratio"]  # [B, T]
            attention_mask = (x == 0)

            z_i = self.forward(x, attention_mask)

            pred_ratio = self.composition_head(z_i)

            loss = self.reg_loss(pred_ratio, target_ratio)

            self.log("val_mse", loss, on_epoch=True, prog_bar=True)
            return loss

        elif self.task == 'panel_expansion_continuous':
            x = batch['x']
            continuous = batch['continuous']
            cont_mask = batch['continuous_mask']
            attention_mask = (x == 0)

            logits = self.forward(x, attention_mask)  # [B, L, num_proteins]
            # logits = logits.mean(dim=1)
            # loss = F.mse_loss(logits[cont_mask], continuous[cont_mask])
            continuous_exp = continuous.unsqueeze(1).expand(-1, logits.size(1), -1)  # [B, L, P]
            mask_exp = cont_mask.unsqueeze(1).expand(-1, logits.size(1), -1)       # [B, L, P]

            loss = F.mse_loss(logits[mask_exp], continuous_exp[mask_exp])
            # loss = F.l1_loss(logits[mask_exp], continuous_exp[mask_exp])
            rmse = torch.sqrt(F.mse_loss(logits[mask_exp], continuous_exp[mask_exp]))

            self._val_cont_preds.append(logits.detach())
            # self._val_cont_targets.append(continuous.detach())
            # self._val_cont_masks.append(cont_mask)
            self._val_cont_targets.append(continuous_exp.detach())
            self._val_cont_masks.append(mask_exp)

            return loss

        elif self.task == 'panel_expansion_continuous_new':
        
            x = batch['x']
            continuous = batch['continuous']
            cont_mask = batch['continuous_mask']
            bin_mat = batch['bin_mat']
            attention_mask = (x == 0)

            logits = self.forward(x, attention_mask)  # [B, P]
            
            self._val_cont_preds.append(logits.detach())
            self._val_cont_targets.append(continuous.detach())
            self._val_cont_masks.append(cont_mask)

            return None
        
        elif self.task == 'image_integration':
            x = batch['x']
            labels = batch['labels']
            HE_embedding = batch['HE_embedding']
            graph_PE = batch['graph_PE']
            attention_mask = (x == 0)
            cell_embedding = self.forward(x, attention_mask)
            image_embedding = self.img_head(HE_embedding)
            g_pe = self.PE_mlp(graph_PE)
            cell_embedding = F.normalize(cell_embedding, p=2, dim=-1)
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)
            # image_embedding = image_embedding + 0.1 * torch.randn_like(image_embedding)
            g_pe = F.normalize(g_pe, p=2, dim=-1)
            temp = 1
            # Query: cell_embedding, Key/Value: image_embedding
            attn_scores = torch.matmul(cell_embedding, image_embedding.T) / temp
            attn_weights = torch.softmax(attn_scores, dim=-1)
            guidance = torch.matmul(attn_weights, image_embedding)
            alpha = 0.1
            beta = 0.1
            fused_embedding = cell_embedding + guidance * alpha + g_pe * beta
            # fused_input = torch.cat([cell_embedding, image_embedding, g_pe], dim=-1)
            # fused_embedding = self.fusion_mlp(fused_input)
            fused_embedding = F.normalize(fused_embedding, p=2, dim=-1)
            pred_features = self.decoder(fused_embedding)
            emb_dist_fused = torch.cdist(fused_embedding, fused_embedding, p=2)
            emb_dist_vit = torch.cdist(image_embedding, image_embedding, p=2)
            loss_geom = ((emb_dist_fused - emb_dist_vit)**2).mean()
            # loss = self.reg_loss(pred_features, torch.cat([cell_embedding, HE_embedding], dim=-1)) + 0.1 * loss_spatial
            loss = self.reg_loss(pred_features, cell_embedding) + 0.1 * loss_geom
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

            return loss

        elif self.task == 'reconstruction':
            x = batch['x']
            with torch.no_grad():
                masked_indices, mask, attention_mask, real_indices = complete_masking(
                    x,
                    n_tokens=self.n_tokens
                )

            embedding, transformer_output = self.forward(masked_indices, attention_mask)
            mlm_logits = self.mlm_head(transformer_output)
            targets = torch.where(mask == 1, real_indices, torch.tensor(-100, device=real_indices.device))
            loss = F.cross_entropy(mlm_logits.view(-1, self.n_tokens), targets.view(-1), ignore_index=-100)
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            return loss
        
        
        elif self.task == 'label_transfer':
            x = batch['x']
            labels = batch['labels']
            # continuous = batch['continuous']
            attention_mask = (x == 0)
            embedding, transformer_output = self.forward(x, attention_mask)
            logits = self.label_classifier(embedding)
            loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            self.val_f1.update(preds, labels)
            self.val_acc.update(preds, labels)
            self.log("val_f1", self.val_f1, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val_acc", self.val_acc, on_epoch=True, on_step=False, prog_bar=True)

            return loss

    def on_train_epoch_end(self):
        if self.task == 'panel_expansion_continuous':
            all_preds = torch.cat(self._train_cont_preds, dim=0)
            all_targets = torch.cat(self._train_cont_targets, dim=0)
            all_masks = torch.cat(self._train_cont_masks, dim=0)
            preds_masked = all_preds[all_masks].cpu().numpy().flatten()
            targets_masked = all_targets[all_masks].cpu().numpy().flatten()

            # RMSE
            mse = np.mean((preds_masked - targets_masked) ** 2)
            rmse = np.sqrt(mse)

            # Spearman
            spearman_list = []

            for p in range(all_preds.shape[1]):
                mask_p = all_masks[:, p]
                if mask_p.sum() < 2:
                    continue

                pred_p = all_preds[mask_p, p].cpu().numpy()
                tgt_p = all_targets[mask_p, p].cpu().numpy()

                sp = spearmanr(pred_p, tgt_p)[0]
                if not np.isnan(sp):
                    spearman_list.append(sp)

            spearman = float(np.mean(spearman_list)) if len(spearman_list) > 0 else 0.0
            self.log("train_spearman_epoch", spearman, prog_bar=True)

            self._train_cont_preds = []
            self._train_cont_targets = []
            self._train_cont_masks = []
        
            self._train_cont_preds = []
            self._train_cont_targets = []
            self._train_cont_masks = []
        
    def on_validation_epoch_end(self):
        if self.task == 'panel_expansion_continuous':
            all_preds = torch.cat(self._val_cont_preds, dim=0)
            all_targets = torch.cat(self._val_cont_targets, dim=0)
            all_masks = torch.cat(self._val_cont_masks, dim=0)

            preds_masked = all_preds[all_masks].cpu().numpy().flatten()
            targets_masked = all_targets[all_masks].cpu().numpy().flatten()
    
            mse = np.mean((preds_masked - targets_masked) ** 2)
            rmse = np.sqrt(mse)
    
            spearman_list = []
            pearson_list = []
    
            for p in range(all_preds.shape[1]):
                mask_p = all_masks[:, p]
                if mask_p.sum() < 2:
                    continue
                
                pred_p = all_preds[mask_p, p].cpu().numpy()
                tgt_p = all_targets[mask_p, p].cpu().numpy()
    
                sp = spearmanr(pred_p, tgt_p)[0]
                if not np.isnan(sp):
                    spearman_list.append(sp)
    
                pr = pearsonr(pred_p, tgt_p)[0]
                if not np.isnan(pr):
                    pearson_list.append(pr)
    
            spearman = float(np.mean(spearman_list)) if len(spearman_list) > 0 else 0.0
            pearson = float(np.mean(pearson_list)) if len(pearson_list) > 0 else 0.0
            cosine = np.dot(preds_masked, targets_masked) / (
                np.linalg.norm(preds_masked) * np.linalg.norm(targets_masked)
            )
    
            self.log("val_rmse_epoch", rmse, prog_bar=True)
            self.log("val_spearman_epoch", spearman, prog_bar=True)
            self.log("val_pearson_epoch", pearson, prog_bar=True)
            self.log("val_cosine_epoch", cosine, prog_bar=True)

            self._val_cont_preds = []
            self._val_cont_targets = []
            self._val_cont_masks = []
        elif self.task == 'panel_expansion_continuous_new':

            all_preds = torch.cat(self._val_cont_preds, dim=0)     # [N, P]
            all_targets = torch.cat(self._val_cont_targets, dim=0) # [N, P]
            all_masks = torch.cat(self._val_cont_masks, dim=0)     # [N, P]

            preds_masked = all_preds[all_masks].cpu().numpy()
            targets_masked = all_targets[all_masks].cpu().numpy()

            mse = np.mean((preds_masked - targets_masked) ** 2)
            rmse = np.sqrt(mse)

            spearman_list = []
            pearson_list = []

            P = all_preds.shape[1]

            for p in range(P):
                mask_p = all_masks[:, p]

                if mask_p.sum() < 2:
                    continue

                pred_p = all_preds[mask_p, p].cpu().numpy()
                tgt_p = all_targets[mask_p, p].cpu().numpy()

                # --- Spearman ---
                sp = spearmanr(pred_p, tgt_p)[0]
                if not np.isnan(sp):
                    spearman_list.append(sp)

                # --- Pearson ---
                pr = pearsonr(pred_p, tgt_p)[0]
                if not np.isnan(pr):
                    pearson_list.append(pr)

            spearman = float(np.mean(spearman_list)) if len(spearman_list) > 0 else 0.0
            pearson = float(np.mean(pearson_list)) if len(pearson_list) > 0 else 0.0

            cell_cosines = []

            for i in range(all_preds.shape[0]):
                mask_i = all_masks[i]

                if mask_i.sum() < 2:
                    continue
                
                pred_i = all_preds[i][mask_i].cpu().numpy()
                tgt_i = all_targets[i][mask_i].cpu().numpy()

                cos_i = np.dot(pred_i, tgt_i) / (
                    np.linalg.norm(pred_i) * np.linalg.norm(tgt_i) + 1e-8
                )

                cell_cosines.append(cos_i)

            cosine = float(np.mean(cell_cosines)) if len(cell_cosines) > 0 else 0.0

            cell_spearman_list = []

            for i in range(all_preds.shape[0]):
                mask_i = all_masks[i]

                if mask_i.sum() < 2:
                    continue
                
                pred_i = all_preds[i][mask_i].cpu().numpy()
                tgt_i = all_targets[i][mask_i].cpu().numpy()

                sp = spearmanr(pred_i, tgt_i)[0]

                if not np.isnan(sp):
                    cell_spearman_list.append(sp)

            cell_spearman = float(np.mean(cell_spearman_list)) if len(cell_spearman_list) > 0 else 0.0

            self.log("val_pearson_epoch", pearson, prog_bar=True)
            # self.log("val_cosine_epoch", cosine, prog_bar=True)
            self.log("val_cell_spearman_epoch", cell_spearman, prog_bar=True)

            self._val_cont_preds = []
            self._val_cont_targets = []
            self._val_cont_masks = []
            
        else:
            pass

    def get_embedding(self, x, attention_mask=None):
        token_embeddings = self.embeddings(x)
        if self.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embeddings.device))
            embeddings = token_embeddings + pos_embedding
        else:
            embeddings = self.positional_embedding(token_embeddings)

        embeddings = self.layernorm_input(embeddings)
        embeddings = self.dropout(embeddings)

        transformer_output = self.encoder(
            embeddings, src_key_padding_mask=attention_mask
        )

        cls_emb = transformer_output[:, 0]   # [B, D]
        return cls_emb

    def configure_optimizers(self):
        optimizer_params = []
        if self.task == "cell_type_prediction":
            mode = self.finetune_mode
            if mode == "linear_probe":
                optimizer_params.append({"params": self.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            elif mode == "partial":
                if hasattr(self.encoder, "layers"):
                    unfrozen_encoder_params = list(self.encoder.layers[-4:].parameters())
                    if unfrozen_encoder_params:
                        optimizer_params.append({"params": unfrozen_encoder_params, "lr": 1e-4, "weight_decay": 0.01})
                optimizer_params.append({"params": self.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            elif mode == "full":
                optimizer_params.append({"params": self.encoder.parameters(), "lr": 5e-4, "weight_decay": 0.01})
                optimizer_params.append({"params": self.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01})
        elif self.task == 'Prototype_classification':
            mode = self.finetune_mode
            if mode == "linear_probe":
                optimizer_params.append({"params": self.prototypes, "lr": 1e-3, "weight_decay": 0.01})
            elif mode == "partial":
                if hasattr(self.encoder, "layers"):
                    unfrozen_encoder_params = list(self.encoder.layers[-4:].parameters())
                    if unfrozen_encoder_params:
                        optimizer_params.append({"params": unfrozen_encoder_params, "lr": 1e-4, "weight_decay": 0.01})
                optimizer_params.append({"params": self.prototypes, "lr": 1e-3, "weight_decay": 0.01})
            elif mode == "full":
                optimizer_params.append({"params": self.encoder.parameters(), "lr": 1e-4, "weight_decay": 0.01})
                optimizer_params.append({"params": self.prototypes, "lr": 1e-3, "weight_decay": 0.01})
            else:
                raise ValueError(f"Unknown finetune_mode '{mode}'")
        elif self.task == "neighborhood_identify":
            optimizer_params.append({"params": self.encoder.parameters(), "lr": 1e-4, "weight_decay": 0.1})
            optimizer_params.append({"params": self.embeddings.parameters(), "lr": 1e-4, "weight_decay": 0.1})
        elif self.task == "panel_expansion_continuous_new":
            optimizer_params.append({"params": self.encoder.parameters(), "lr": 1e-4, "weight_decay": 0.1})
            optimizer_params.append({"params": self.embeddings.parameters(), "lr": 1e-4, "weight_decay": 0.1})
            optimizer_params.append({"params": self.decoder.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.proj.parameters(), "lr": 1e-3, "weight_decay": 0.01})

        elif self.task == "image_integration":
            optimizer_params.append({"params": self.encoder.parameters(), "lr": 1e-4, "weight_decay": 0.1})
            optimizer_params.append({"params": self.embeddings.parameters(), "lr": 1e-4, "weight_decay": 0.1})
            optimizer_params.append({"params": self.img_head.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.PE_mlp.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.fusion_mlp.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.decoder.parameters(), "lr": 1e-3, "weight_decay": 0.01})
        elif self.task == "pan_cancer_engine":
            optimizer_params.append({"params": self.encoder.parameters(), "lr": 1e-4, "weight_decay": 0.01})
            optimizer_params.append({"params": self.mlm_head.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.shared_proj.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.private_proj.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.private_classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.domain_classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01})
        elif self.task == 'reconstruction':
            optimizer_params.append({"params": self.encoder.parameters(), "lr": 1e-4, "weight_decay": 0.01})
            optimizer_params.append({"params": self.mlm_head.parameters(), "lr": 1e-3, "weight_decay": 0.01})
        elif self.task == 'label_transfer':
            optimizer_params.append({"params": self.proj.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.encoder.parameters(), "lr": 1e-3, "weight_decay": 0.01})
            optimizer_params.append({"params": self.label_classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01}) 
        else:
            raise ValueError(f"Unknown finetune_mode '{mode}'")
        optimizer = torch.optim.AdamW(optimizer_params)
        return optimizer
