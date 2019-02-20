import torch
import numpy as np
import time

def tile(a, dim, n_tile):
    """Expands a tensor amongst a given dimension, repeating its components."""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    if a.is_cuda:
        order_index = order_index.cuda()
    return torch.index_select(a, dim, order_index)

def now_str():
    return time.strftime("%d-%m-%Y_%Hh%Mm%S")