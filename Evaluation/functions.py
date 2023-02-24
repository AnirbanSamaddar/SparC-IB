import torch
import os
import sys
import numpy as np
import math


def brier_score(logits,y):
    tmp = logits.clone()           
    tmp = tmp.reshape((logits.shape[0]*logits.shape[1],logits.shape[2]))
    y = y.repeat(logits.shape[0])            
    logits = tmp
    a = torch.nn.functional.one_hot(y,num_classes=10).squeeze()
    logits_prob = torch.exp(logits)
    logits_prob[torch.isinf(logits_prob)] = 1e7
    logits_prob = logits_prob/logits_prob.sum(dim=1)[:,None]
    brier = ((logits_prob * a - a)**2).sum(dim=1).mean(dim=0)
    return brier


