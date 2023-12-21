import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import apex
from utils import *

def build_head(config):
    name = config.MODEL.head.name
    if name=='Arc_face':
        head = ArcMarginProduct(config.MODEL.Embedding_dim, config.MODEL.num_classes)
    elif name=='Sub_Arc_face':
        head = ArcMarginProduct_subcenter(config.MODEL.Embedding_dim, config.MODEL.num_classes, k=config.MODEL.head.K)
    else:
        raise NotImplementedError(f"Unkown head: {name}")
    return head
    
## Used Model
class clip_vit(nn.Module):
    def __init__(self, config, logger):
        super(clip_vit, self).__init__()        
        self.model = VisionTransformer(
                  image_size=config.MODEL.backbone.VIT.image_size,
                  patch_size=config.MODEL.backbone.VIT.patch_size,
                  width=config.MODEL.backbone.VIT.width,
                  layers=config.MODEL.backbone.VIT.layers,
                  heads=config.MODEL.backbone.VIT.heads,
                  mlp_ratio=config.MODEL.backbone.VIT.mlp_ratio,
                  output_dim=config.MODEL.backbone.VIT.output_dim
              )
        self.drop = nn.Dropout(0.2)
        self.proj = nn.Linear(1024, 64)
        self.cont_head = build_head(config)
        self.model.load_state_dict(torch.load(config.MODEL.backbone.pretrained), strict=True)
        logger.info(f"=> Load pretrained vit_backbone '{config.MODEL.backbone.pretrained}' successfully")
        for param in self.model.parameters():
            param.requires_grad=False
    
    def forward_embedding(self, x):
        x = self.model(x)
        x = self.drop(x)
        x = self.proj(x)
        return x
    
    def forward(self, x):
        x = self.model(x)
        x = self.drop(x)
        x = self.proj(x)
        logits = self.cont_head(x)
        return logits