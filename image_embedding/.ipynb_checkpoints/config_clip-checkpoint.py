# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.IMG_SIZE = 224
_C.SEED = 3407
# -----------------------------------------------------------------------------
# Model settings 126827
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'CLIP_Vit'
_C.MODEL.Embedding_dim = 64
_C.MODEL.num_classes = 27704
_C.MODEL.finetune = None
_C.MODEL.output_dir = '/root/autodl-tmp/vit_Gaint'

#head
_C.MODEL.head = CN()
_C.MODEL.head.name = 'Arc_face'

#CLIP-H-16
_C.MODEL.backbone = CN()
_C.MODEL.backbone.pretrained = 'pretrained_models/ViT_G_14_2B_vision_model.pt'
_C.MODEL.backbone.VIT = CN()
_C.MODEL.backbone.VIT.image_size = 224
_C.MODEL.backbone.VIT.patch_size = 14
_C.MODEL.backbone.VIT.width = 1408
_C.MODEL.backbone.VIT.layers = 40
_C.MODEL.backbone.VIT.heads = 16
_C.MODEL.backbone.VIT.mlp_ratio = 4.3637
_C.MODEL.backbone.VIT.output_dim = 1024

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.weight_decay = 1e-5

def get_config():
    config = _C.clone()
    return config