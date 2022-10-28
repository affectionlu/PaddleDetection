from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

import random
import numpy as np
import os 
import paddle

def torch_pruning_lite(weight,pr,tn):
    n_patch_num = int((weight.shape[1]-1)/tn+1)
    mask=np.ones_like(weight)
    for j in range(n_patch_num):
        n_num=tn if (j+1)*tn<weight.shape[1] else weight.shape[1]-j*tn
        patch=weight[:,j*tn:j*tn+n_num,:,:]
        if(patch.shape[1]<int(tn*pr)):
            mask_patch=np.ones_like(patch)
        else:
            weight_patch=patch.transpose(0,2,3,1)
            mask_patch=np.ones_like(weight_patch)
            rank=np.argsort(np.abs(weight_patch), axis=3)
            for p in range(weight_patch.shape[0]):
                for k in range(weight_patch.shape[1]):
                    for l in range(weight_patch.shape[2]):
                        mask_patch[p][k][l][rank[p][k][l][:int(pr*tn)]]=0
            mask_patch=mask_patch.transpose(0,3,1,2)
        mask[:,j*tn:j*tn+n_num,:,:]=mask_patch
    return mask

def paddle_pruning_phase2(weight,block,pruning_ratio):
    tn=block
    return torch_pruning_lite(weight.numpy(),pruning_ratio,tn)
def paddle_pruning_phase1(weight,block,pruning_ratio):
    tn=weight.shape[1]
    if(tn<block):
        tn=block
    return torch_pruning_lite(weight.numpy(),pruning_ratio,tn)
def my_pruner(model,phase,pruning_ratio):
    prune_mask_dict = {}
    block=16
    for name, layer in model.named_sublayers():
        if isinstance(layer, paddle.nn.quant.quant_layers.QuantizedConv2D) or isinstance(layer, paddle.nn.Conv2D):
            mask_name = layer.parameters()[0].name
            print(mask_name)
            if(phase==1):
                prune_mask_dict[mask_name]=paddle_pruning_phase1(layer.weight,block, pruning_ratio)
            elif(phase==2):
                prune_mask_dict[mask_name]=paddle_pruning_phase2(layer.weight,block, pruning_ratio)
            # print(name,prune_mask_dict[name])
    return prune_mask_dict