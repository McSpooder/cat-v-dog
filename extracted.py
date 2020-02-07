# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Cat v Dog CNN

# %%
#General
from pathlib import PosixPath
import os
import sys


#Data loading
import numpy as np
from PIL import Image


#Model libraries
import fastai.vision as faiv
import matplotlib.pyplot as plt

import torchvision
import torch
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm


# %%
data_dir = "../../data/dog vs cat"


# %%
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on: ", dev)
device = torch.device(dev)

# %% [markdown]
# # Data Block API

# %%
#class BasicImage(faiv.ItemBase):
#     def __init__(self, path):
#         self.data = open_jpg(path)
        
# class BasicImageList(faiv.ItemList):
#     def __init__(self, items, itemsB=None, **kwargs):
#         super().__init__(items, **kwargs)
#         self.itemsB = itemsB
#         self.copy_new.append('itemsB')
        
#     def get(self, i):
#         path = super().get(i)
#         return BasicImage(path)


# %%
def open_jpg(path, size):
    i = Image.open(path,)
    i2 = i.resize(size)
    a = np.asarray(i2) / 255
    t = torch.Tensor(a)
    t2 = t.permute(2,0,1)
    return faiv.Image(t2)
    
class MyImageItemList(faiv.ItemList):
    
    def __init__(self, items, itemsB=None, size=(128,128), **kwargs):
        super().__init__(items, **kwargs)
        self.itemsB = itemsB
        self.copy_new.append('itemsB')
        self.size = size
        
    def get(self, i):
        path = super().get(i)
        return open_jpg(path, self.size)
        
    """ custom item list for nifti files """
    def open(self, fn:faiv.PathOrStr)->faiv.Image: return open_jpg(fn)
    

def get_y_label(x):
    name = x.stem
    label = ""
    for c in name:
        if c != '.':
            label += c
        else:
            break
    return label

# %% [markdown]
# # CNN

# %%
def conv2d(ni:int, nf:int, ks:int=3, stride:int=2, pad:int=1, norm='batch'):
    bias = not norm == 'batch'
    conv = faiv.init_default(nn.Conv2d(ni,nf,ks,stride,pad,bias=bias))
    conv = spectral_norm(conv) if norm == 'spectral' else weight_norm(conv) if norm == 'weight' else conv
    layers = [conv]
    layers += [nn.ReLU(inplace=True)]  # use inplace due to memory constraints
    layers += [nn.BatchNorm2d(nf)] if norm == 'batch' else []
    return nn.Sequential(*layers)

def res2d_block(ni, nf, ks=3, norm='batch', dense=False):
    """ 2d Resnet block of `nf` features """
    return faiv.SequentialEx(conv2d(ni, nf, ks, pad=ks//2, norm=norm),
                             conv2d(nf, nf, ks, pad=ks//2, norm=norm))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# %% [markdown]
# # Playing with Data 

# %%
norm = 'batch'
layers = ([res2d_block(1,16,3,norm=norm)]+
        [res2d_block(16,16,norm=norm) for _ in range(4)]+
        [conv2d(16,1,ks=1,pad=0,norm=None)]+
        [Flatten()]+
        [nn.Linear(10813440, 4)])
model = nn.Sequential(*layers)


# %%
data_dir = PosixPath('../../data/dog vs cat')
data_dir.ls()


# %%
idb = (MyImageItemList.from_folder(data_dir, extensions=('.jpg'))
                    .split_by_folder()
                    .label_from_folder()
                    .databunch(bs=5))

loss = nn.CrossEntropyLoss()
learner = faiv.Learner(idb, model, loss_func=loss)


# %%
idb


# %%
idb.train_ds.x[7000]

# %% [markdown]
# # Training

# %%
learner.summary()


# %%


