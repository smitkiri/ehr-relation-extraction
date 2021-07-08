import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

# import spacy

USE_GPU = False
print("GPU ", USE_GPU)


def to_gpu(x):
    """puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. """
    return x.cuda() if USE_GPU else x


def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c) > 0:
        for l in c: apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


def save_model(m, p):
    print("SAVE MODEL PATH: ", p)
    torch.save(m.state_dict(), p)


def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor.
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else:
            raise NotImplementedError(a.dtype)
    if cuda: a = to_gpu(a)
    return a


def load_ner_model(m, p, strict=True):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()):  # list "detatches" the iterator
        if n not in names and n + '_raw' in names:
            if n + '_raw' not in sd: sd[n + '_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd, strict=strict)
