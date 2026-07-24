import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import math
import numba
import re
from scipy.sparse import issparse
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union
import pyarrow
import pyarrow.parquet as pq
import warnings
from constants import (
    PAD_TOKEN,
    CLS_TOKEN,
    PROTEIN_TOKEN_BASE,
    side_maps,
    protein_id_map,
    SIDE_COLUMNS,
    all_proteins,
    Task,
)
from pathlib import Path

SRC_DIR = Path(__file__).parent

class FineTuneTokenizer:
    def __init__(
        self,
        all_proteins: List[str],
        protein_id_map: dict,
        side_maps: dict,
        protein_token_base: int = PROTEIN_TOKEN_BASE,
        pad_token: int = PAD_TOKEN,
        cls_token: int = CLS_TOKEN,
        protein_mapping_path: str = str(SRC_DIR / "protein_standard_mapping.tsv"),
    ):
        self.all_proteins = all_proteins
        self.protein_id_map = protein_id_map
        self.side_maps = side_maps
        self.protein_token_base = protein_token_base
        self.pad_token = pad_token
        self.cls_token = cls_token

        self.all_proteins_norm = set(
            self._normalize_protein_name(p) for p in self.all_proteins
        )

        # ===== protein standard mapping =====
        self.protein_standard_map = self._load_protein_standard_mapping(
            protein_mapping_path
        )

    @staticmethod
    def _normalize_protein_name(name: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", name.upper())

    def _load_protein_standard_mapping(self, path: str) -> Dict[str, str]:
        """
        return:
            normalized_original_name -> normalized_unified_name
        """
        df = pd.read_csv(path, sep="\t")

        mapping = {}
        for _, row in df.iterrows():
            orig = self._normalize_protein_name(str(row["original_name"]))
            unified = self._normalize_protein_name(str(row["unified_name"]))
            mapping[orig] = unified

        return mapping

    def build_raw_varname_to_tokenid(self, adata_paths):
        """
        raw adata.var_names -> final token_id
        """

        all_proteins_idx = {p: i for i, p in enumerate(self.all_proteins)}

        raw_to_token = {}

        for path, var_list in zip(adata_paths, self.varnames_list):
            for raw_name in var_list:

                # 1. normalize
                p_norm = self._normalize_protein_name(raw_name)

                # 2. standard mapping
                if p_norm in self.protein_standard_map:
                    p_norm = self.protein_standard_map[p_norm]
                if p_norm not in all_proteins_idx:
                    continue

                idx = all_proteins_idx[p_norm]
                token_id = self.protein_token_base + idx

                raw_to_token[raw_name] = token_id

        return raw_to_token

    def collect_data(
        self,
        adata_paths: List[str],
        cell_type_col: Optional[str] = None,
        spatial_key: Optional[str] = None,
    ):
        X_list = []
        varnames_list = []
        side_information = []
        cell_type_list = []
        obs_list = []

        for path in tqdm(adata_paths, desc="Collecting adata"):
            adata = sc.read_h5ad(path)
            mapped_var_names = []
            for p in adata.var_names:
                p_norm = self._normalize_protein_name(p)
                if p_norm in self.protein_standard_map:
                    mapped_var_names.append(self.protein_standard_map[p_norm])
                else:
                    mapped_var_names.append(p_norm)

            adata.var_names = mapped_var_names

            mask = [v in self.all_proteins_norm for v in adata.var_names]

            if not all(mask):
                missing = [
                    adata.var_names[i]
                    for i, ok in enumerate(mask)
                    if not ok
                ]
                warnings.warn(
                    f"{path} has proteins not in global list. "
                    f"They will be removed: {missing}",
                    UserWarning,
                )

            adata = adata[:, mask]

            X_list.append(adata.X)
            varnames_list.append(list(adata.var_names))
            side_information.append(adata.obs[SIDE_COLUMNS].copy())
            obs_list.append(adata.obs.copy())

            if cell_type_col is not None and cell_type_col in adata.obs.columns:
                ct_vals = adata.obs[cell_type_col].astype("category")
                labels = ct_vals.cat.codes.values.astype(np.int32)
                cell_type_list.append(labels)
            else:
                cell_type_list.append(None)

        self.X_list = X_list
        self.varnames_list = varnames_list
        self.side_information = side_information
        self.cell_type_list = cell_type_list
        self.obs_list = obs_list
        self.adata = adata

    def tokenize(self):

        side_columns = SIDE_COLUMNS
        P = len(self.all_proteins)
        S = len(side_columns)

        all_proteins_idx = {p: i for i, p in enumerate(self.all_proteins)}

        side_array_list = []
        for df in self.side_information:
            arr = np.zeros((df.shape[0], S), dtype=np.int32)
            for k, col in enumerate(side_columns):
                mapping = self.side_maps[col]
                arr[:, k] = np.array([mapping.get(str(v), self.pad_token) for v in df[col].values], dtype=np.int32)
            side_array_list.append(arr)

        # numba tokenizer
        @numba.njit(parallel=True)
        def _protein_id_tokenize_numba(X_list_cell, varnames_idx, side_arr, P, S,
                                       PAD_TOKEN, CLS_TOKEN, PROTEIN_TOKEN_BASE):
            n_cells = X_list_cell.shape[0]
            out = np.zeros((n_cells, 1 + S + P), dtype=np.int32)
            out[:, 0] = CLS_TOKEN
            out[:, 1:1+S] = side_arr
            for i in numba.prange(n_cells):
                expr_values = np.full(P, np.nan, dtype=np.float32)
                cell = X_list_cell[i]
                for j in range(len(varnames_idx)):
                    idx = varnames_idx[j]
                    val = cell[j]
                    if not np.isnan(val):
                        expr_values[idx] = val
                nonnan_idx = np.where(~np.isnan(expr_values))[0]
                if len(nonnan_idx) > 0:
                    sorted_idx = nonnan_idx[np.argsort(-expr_values[nonnan_idx])]
                    protein_tokens = sorted_idx + PROTEIN_TOKEN_BASE
                    out[i, 1+S:1+S+len(sorted_idx)] = protein_tokens
            return out

        tokenized_list = []
        for X_mat, varnames, side_arr in tqdm(zip(self.X_list, self.varnames_list, side_array_list), total=len(self.X_list)):
            if hasattr(X_mat, "toarray"):
                X_mat = X_mat.toarray()
            else:
                X_mat = np.array(X_mat)

            varnames_idx = np.array([all_proteins_idx[p] if p in all_proteins_idx else -1 for p in varnames], dtype=np.int32)
            mask = varnames_idx != -1
            X_mat = X_mat[:, mask]
            varnames_idx = varnames_idx[mask]

            tokens = _protein_id_tokenize_numba(X_mat, varnames_idx, side_arr, P, S,
                                                self.pad_token, self.cls_token, self.protein_token_base)
            tokenized_list.append(tokens)

        return tokenized_list