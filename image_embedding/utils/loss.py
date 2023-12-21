import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Smoth_CE_Loss(nn.Module):
    def __init__(self, ls_=0.9):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(reduction="none")  
        self.ls_ = ls_

    def forward(self, logits, labels):
        labels *= self.ls_
        return self.crit(logits, labels)

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=1)
        loss = -logprobs * target
        loss = loss.sum(dim=1)
        return loss

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if crit == "ce":
            self.crit = DenseCrossEntropy()
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.reduction = reduction
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        labels = F.one_hot(labels.long(), logits.shape[-1]).float()
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss