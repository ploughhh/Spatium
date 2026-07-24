import os
import numpy as np
from math import ceil
from os.path import join
from typing import Dict, List
import torch
from torch.distributed import get_rank
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset, BatchSampler
import pyarrow.parquet as pq
import pytorch_lightning as pl
import merlin.io
from merlin.dataloader.torch import Loader
from merlin.dtypes import boolean
from merlin.dtypes import float32, int64, int32, string
from merlin.schema import ColumnSchema, Schema
import pyarrow.dataset as ds
import zarr
from tokenizer import FineTuneTokenizer
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import random
from collections import defaultdict



def write_zarr_dataset(task: str,
                       save_dir: str,
                       tokenizer: FineTuneTokenizer,
                       adata=None,
                       k=30):
    os.makedirs(save_dir, exist_ok=True)
    zarr_path = save_dir
    root = zarr.open(zarr_path, mode='w')
    tokenized_list = tokenizer.tokenize()
    token_dim = tokenized_list[0].shape[1]
    tokens_z = root.create_dataset(
        name='tokens',
        shape=(0, token_dim),
        chunks=(10000, token_dim),
        dtype='int64',
        maxshape=(None, token_dim)
    )
    if task == 'cell_type_prediction' or 'Prototype_classification':
        label_z = root.create_dataset(
            name='label',
            shape=(0,),
            chunks=(10000,),
            dtype='int64',
            maxshape=(None,)
        )
        for tokens, neigh in tqdm(
            zip(tokenized_list, tokenizer.cell_type_list),
            total=len(tokenized_list),
            desc="Writing Zarr (tokens + label)"
        ):
            # tokens: [n_cells, token_dim]
            # neigh:  [n_cells] / [n_cells, 1]

            tokens_z.append(tokens.astype(np.int64))
            label_z.append(neigh.reshape(-1).astype(np.int64))

        print("Zarr saved.")
        print("tokens shape:", root['tokens'].shape)
        print("label shape:", root['label'].shape)
    elif task == 'neighborhood_identify':
        assert adata is not None, \
            "Task 'neighborhood_identify' requires `adata`, but got None."
        assert 'spatial' in adata.obsm, \
            "Task 'neighborhood_identify' requires `adata.obsm['spatial']`, but it was not found."
        cell_types = tokenizer.cell_type_list[0]  # [n_cells,]
        coords = adata.obsm['spatial']            # [n_cells, 2]
        num_types = cell_types.max() + 1

        one_hot_types = np.eye(num_types)[cell_types]  # [n_cells, num_types]

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        neighbor_indices = indices[:, 1:]  # [n_cells, k]

        neighbor_one_hot = one_hot_types[neighbor_indices]  # [n_cells, k, num_types]
        neighbor_sum = neighbor_one_hot.sum(axis=1)         # [n_cells, num_types]

        neighbor_ratio = neighbor_sum / neighbor_sum.sum(axis=1, keepdims=True)  # [n_cells, num_types]
        neighbor_ratio_list = [neighbor_ratio]
        num_types = neighbor_ratio_list[0].shape[1]

        neighbor_ratio_z = root.create_dataset(
            name='neighbor_ratio',
            shape=(0, tokenizer.num_types),
            chunks=(10000, tokenizer.num_types),
            dtype='float32',
            maxshape=(None, tokenizer.num_types)
        )
        for tokens, neigh in tqdm(
            zip(tokenized_list, neighbor_ratio_list),
            total=len(tokenized_list),
            desc="Writing Zarr (tokens + neighbor_ratio)"
        ):

            tokens_z.append(tokens.astype(np.int64))
            neighbor_ratio_z.append(neigh.astype(np.float32))

        print("Zarr saved.")
        print("tokens shape:", root['tokens'].shape)
        print("neighbor_ratio shape:", root['neighbor_ratio'].shape)





class TokenZarrDataset(Dataset):
    def __init__(self, zarr_path, task=None, token_dim=None):
        self.root = zarr.open(zarr_path, mode='r')
        self.token = self.root['tokens'][:]
        self.label = self.root.get('label', None)
        if self.label is not None:
            self.label = self.root['label'][:]
        self.neighbor_ratio = self.root.get('neighbor_ratio', None)
        self.mask = self.root.get('mask', None)
        self.continuous = self.root.get('continuous', None)
        self.continuous_mask = self.root.get('continuous_mask', None)
        self.bin_mat = self.root.get('bin_mat', None)
        self.HE_embedding = self.root.get('HE_embedding', None)
        self.graph_PE = self.root.get('graph_PE', None)

        self.task = task
        self.token_dim = token_dim
        

        if task == 'pan_cancer_engine':
            self.dataset_id = self.token[:, 2]
            self.disease_type = self.token[:, 4]
            self.idx_by_group = {}
            for i, (ds, dis) in enumerate(zip(self.dataset_id, self.disease_type)):
                key = (int(ds), int(dis))
                if key not in self.idx_by_group:
                    self.idx_by_group[key] = []
                self.idx_by_group[key].append(i)

    def __len__(self):
        return self.token.shape[0]

    def __getitem__(self, idx):
        # -------- tokens --------
        x = torch.tensor(self.token[idx], dtype=torch.long)

        out = {'x': x}
        if self.task in ['cell_type_prediction', 'Prototype_classification']:
            if self.label is None:
                raise RuntimeError("Cell type <label> path not found in Zarr")
            out['labels'] = torch.tensor(self.label[idx], dtype=torch.long)
        elif self.task == 'neighborhood_identify':
            if self.neighbor_ratio is None:
                raise RuntimeError("Neighbor ratio <neighbor_ratio> path not found in Zarr")
            out['neighbor_ratio'] = torch.tensor(
                self.neighbor_ratio[idx],
                dtype=torch.float32
            )
        elif self.task == 'panel_expansion_continuous_new':
            out['continuous_mask'] = torch.tensor(self.continuous_mask[idx], dtype=torch.bool)
            out['continuous'] = torch.tensor(self.continuous[idx], dtype=torch.float32)
            out['bin_mat'] = torch.tensor(self.bin_mat[idx], dtype=torch.long)
        elif self.task == 'image_integration':
            out['labels'] = torch.tensor(self.label[idx], dtype=torch.long)
            out['HE_embedding'] = torch.tensor(self.HE_embedding[idx], dtype=torch.float32)
            out['graph_PE'] = torch.tensor(self.graph_PE[idx], dtype=torch.float32)
        elif self.task == 'label_transfer':
            out['labels'] = torch.tensor(self.label[idx], dtype=torch.long)
            # out['continuous'] = torch.tensor(self.continuous[idx], dtype=torch.float32)
            # out['bin_mat'] = torch.tensor(self.bin_mat[idx], dtype=torch.long)

        return out


class MerlinDataModule(pl.LightningDataModule):
    # def __init__(self, parquet_files, batch_size=128, num_workers=8, has_label=False, split_by_file=False):
    def __init__(self, zarr_path, batch_size=128, num_workers=8, task=None, split_by_file=False, context_length=None, num_types=None, k=None):
        super().__init__()
        # self.parquet_files = parquet_files
        self.zarr_path = zarr_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.split_by_file = split_by_file
        self.context_length = context_length


    
    def setup(self, stage=None):
        if self.split_by_file and len(self.zarr_path) >= 2:
            train_dataset = TokenZarrDataset(self.zarr_path[0], self.task, self.context_length)
            val_dataset   = TokenZarrDataset(self.zarr_path[1], self.task, self.context_length)
            self.train_dataset = train_dataset
            self.val_dataset   = val_dataset
            self.test_dataset  = torch.utils.data.Subset(torch.utils.data.Dataset(), [])
        else:
            dataset = TokenZarrDataset(self.zarr_path, self.task, self.context_length)
            n = len(dataset)
            n_train = int(n * 0.8)
            n_val   = n - n_train
            n_test  = None

            if self.task == 'pan_cancer_engine':
                self.train_dataset = dataset
                self.val_dataset   = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
                self.test_dataset  = None
            else:
                self.train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
                self.val_dataset   = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
                self.test_dataset  = None

    def train_dataloader(self):
        if self.task == 'cell_type_prediction':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        elif self.task == 'Prototype_classification':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        elif self.task == 'neighborhood_identify':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        elif self.task == 'panel_expansion':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        elif self.task == 'panel_expansion_continuous':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        elif self.task == 'panel_expansion_continuous_new':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        elif self.task == 'image_integration':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        elif self.task == 'reconstruction':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        elif self.task == 'label_transfer':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

            
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)



class FewShotTokenDataset(Dataset):
    """
    Few-shot Dataset
    - support: token + label
    - query: token only
    - task: 'cell_type_prediction' or 'Prototype_classification'
    """
    def __init__(self, zarr_path, task='Prototype_classification', has_label=True):

        self.data = zarr.open(zarr_path, mode='r')
        self.task = task
        self.has_label = has_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        x = row[:-1] if self.has_label else row
        label = row[-1] if self.has_label else -1
        return {
            'x': torch.tensor(x, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MerlinFewShotDataModule(pl.LightningDataModule):

    def __init__(self, support_path, query_path, batch_size=128, num_workers=8, task='Prototype_classification'):
        super().__init__()
        self.support_path = support_path
        self.query_path = query_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task

    def setup(self, stage=None):
        self.train_dataset = FewShotTokenDataset(self.support_path, task=self.task, has_label=True)
        self.val_dataset   = FewShotTokenDataset(self.query_path, task=self.task, has_label=True)
        self.test_dataset  = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


