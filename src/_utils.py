import os
import torch
import random
import numpy as np
from constants import PROTEIN_TOKEN_BASE


def complete_masking(batch_tokens, p=0.15, n_tokens=None):
    """
    batch_tokens: [batch_size, seq_len] int tensor
    p: masking probability
    n_tokens: vocab size
    
    Returns:
        masked_tokens, mask_tensor, attention_mask, original_tokens
    """
    PAD = 0
    CLS = 1
    SIDE_TOKENS = set(range(2,PROTEIN_TOKEN_BASE))
    if n_tokens is None:
        n_tokens = int(X.max().item() + 1)
    MASK_TOKEN = n_tokens-1

    X = batch_tokens.clone()
    original_tokens = X.clone()
    batch_size, seq_len = X.shape

    candidate_mask = (~torch.isin(X, torch.tensor(list(SIDE_TOKENS), device=X.device))) & (X != PAD) & (X != CLS)

    # 15% probability
    mask_bernoulli = torch.bernoulli(torch.full(X.shape, p, device=X.device)).bool()
    mask_positions = mask_bernoulli & candidate_mask

    masked_tokens = X.clone()
    mask_tensor = torch.zeros_like(X, dtype=torch.int32)

    mask_idx = mask_positions.nonzero(as_tuple=False)
    num_mask = mask_idx.shape[0]

    if num_mask > 0:
        perm = torch.randperm(num_mask, device=X.device)
        num_80 = int(num_mask * 0.8)
        num_10 = int(num_mask * 0.1)

        rank_min = max(SIDE_TOKENS) + 1

        # 80% mask -> MASK_TOKEN
        idx_80 = mask_idx[perm[:num_80]]
        masked_tokens[idx_80[:,0], idx_80[:,1]] = MASK_TOKEN
        mask_tensor[idx_80[:,0], idx_80[:,1]] = 1

        # 10% random token
        idx_10 = mask_idx[perm[num_80:num_80+num_10]]
        rand_tokens = torch.randint(rank_min, n_tokens-1, (num_10,), device=X.device)
        masked_tokens[idx_10[:,0], idx_10[:,1]] = rand_tokens
        mask_tensor[idx_10[:,0], idx_10[:,1]] = 1

        # 10% keep original
        idx_10_keep = mask_idx[perm[num_80+num_10:]]
        mask_tensor[idx_10_keep[:,0], idx_10_keep[:,1]] = 1

    masked_tokens[X==PAD] = PAD
    mask_tensor[X==PAD] = 0

    attention_mask = (X == PAD).type(torch.bool)

    return masked_tokens, mask_tensor, attention_mask, original_tokens
