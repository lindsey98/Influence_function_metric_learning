#! /usr/bin/env python3
import torch
from torch.autograd import grad
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import math

@torch.no_grad()
def calc_loss_train_indices(model, dataset, criterion, indices):
    '''
        Calculate all training losses
    '''
    l = []
    model.eval()
    for ind in indices:
        x, t, _= dataset[ind]
        x, t = torch.unsqueeze(x, 0).cuda(), torch.tensor([t]).cuda()
        m = model(x)
        l_this = criterion(m, None, t)
        l.append(l_this.detach().cpu().item())
    return l

@torch.no_grad()
def calc_loss_train(model, dl, criterion, indices=None):
    '''
        Calculate all training losses
    '''
    l = []
    model.eval()
    for ct, (x, t, _) in tqdm(enumerate(dl)):
        x, t = x.cuda(), t.cuda()
        m = model(x)
        l_this = criterion(m, None, t)
        l.append(l_this.detach().cpu().item())
    if indices is not None:
        l = l[indices]
    return l

def loss_change_train(model, criterion, dl_tr, params_prev, params_cur):
    '''
        Calculate training L(params_prev) and L(params_cur)
    '''
    weight_orig = model.module[-1].weight.data # cache original parameters
    model.module[-1].weight.data = params_prev
    l_prev = calc_loss_train(model, dl_tr, criterion, None)

    model.module[-1].weight.data = params_cur
    l_cur = calc_loss_train(model, dl_tr, criterion, None)

    model.module[-1].weight.data = weight_orig # dont forget to revise the weights back to the original
    return np.asarray(l_prev), np.asarray(l_cur)


def calc_inter_dist_pair(feat_cls1, feat_cls2):
    '''
        Calculate d(p_c)
    '''
    feat_cls1 = F.normalize(feat_cls1, p=2, dim=-1) # L2 normalization
    feat_cls2 = F.normalize(feat_cls2, p=2, dim=-1)

    if len(feat_cls1.shape) == 1 and len(feat_cls2.shape) == 1:
        dist = (feat_cls1 - feat_cls2).square().sum()
        return dist
    inter_dis = torch.cdist(feat_cls1, feat_cls2).square()  # inter class distance
    inter_dis = inter_dis.diagonal().sum() # only sum aligned pairs
    return inter_dis

def grad_confusion_pair(model, all_features, wrong_indices, confusion_indices):
    '''
        Calculate  \triangle d(p_c) / \triangle theta
    '''
    cls_features = all_features[wrong_indices]
    confuse_cls_features = all_features[confusion_indices]

    model.zero_grad()
    model.eval()
    cls_features, confuse_cls_features = cls_features.cuda(), confuse_cls_features.cuda()

    feature1 = model.module[-1](cls_features)  # (N', 512)
    feature2 = model.module[-1](confuse_cls_features)  # (N', 512)
    confusion = calc_inter_dist_pair(feature1, feature2)

    params = model.module[-1].weight
    grad_confusion2params = list(grad(confusion, params))
    grad_confusion2params = [y.detach().cpu() for y in grad_confusion2params]  # accumulate gradients
    confusion = confusion.detach().cpu().item()  # accumulate confusion

    return confusion, grad_confusion2params


def avg_confusion(cls, confusion_classes,
                  pred, label, nn_indices):
    '''
        get confusion indices
    '''
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    nn_indices = nn_indices.flatten()
    assert len(pred) == len(label)
    assert len(pred) == len(nn_indices)

    # Get cls samples indices and confusion_classes samples indices
    wrong_indices = [[] for _ in range(len(confusion_classes))]  # belong to cls and wrongly predicted
    confuse_indices = [[] for _ in range(
        len(confusion_classes))]  # belong to confusion classes and are neighbors of interest_indices
    pair_counts = 0  # count how many confusion pairs in total
    for kk, confusion_cls in enumerate(confusion_classes):
        wrong_as_confusion_cls_indices = np.where((pred == confusion_cls) & (label == cls))[0]
        wrong_indices[kk] = wrong_as_confusion_cls_indices
        confuse_indices[kk] = nn_indices[wrong_as_confusion_cls_indices]
        pair_counts += len(wrong_as_confusion_cls_indices)

    return wrong_indices, confuse_indices


def grad_confusion(model, all_features, cls, confusion_classes,
                   pred, label, nn_indices):
    '''
        Calculate  \triangle Avg{d(p_c)} / \triangle theta
    '''
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    nn_indices = nn_indices.flatten()
    assert len(pred) == len(label)
    assert len(pred) == len(nn_indices)

    # Get cls samples indices and confusion_classes samples indices
    wrong_indices = [[] for _ in range(len(confusion_classes))] # belong to cls and wrongly predicted
    confuse_indices = [[] for _ in range(len(confusion_classes))] # belong to confusion classes and are neighbors of interest_indices
    pair_counts = 0 # count how many confusion pairs in total
    for kk, confusion_cls in enumerate(confusion_classes):
        wrong_as_confusion_cls_indices = np.where((pred == confusion_cls) & (label == cls))[0]
        wrong_indices[kk] = wrong_as_confusion_cls_indices
        confuse_indices[kk] = nn_indices[wrong_as_confusion_cls_indices]
        pair_counts += len(wrong_as_confusion_cls_indices)

    # Compute pairwise confusion and record gradients to projection layer's weights
    accum_grads = [torch.zeros_like(model.module[-1].weight).detach().cpu()]
    accum_confusion = 0.
    for kk in range(len(confusion_classes)):
        confusion, grad_confusion2params = grad_confusion_pair(model, all_features,
                                                               wrong_indices[kk],
                                                               confuse_indices[kk])
        accum_grads = [x + y for x, y in zip(accum_grads, grad_confusion2params)] # accumulate gradients
        accum_confusion += confusion # accumulate confusion

    accum_grads = [x / pair_counts for x in accum_grads]
    accum_confusion = accum_confusion / pair_counts
    return accum_confusion, accum_grads

def calc_influential_func_sample(grad_alltrain):
    '''
        Calculate L(theta')-L(theta)
    '''
    l_prev = grad_alltrain['l_prev']
    l_cur = grad_alltrain['l_cur']
    l_diff = np.stack(l_cur) - np.stack(l_prev) # l_diff = l'-l0, if l_diff < 0, helpful, otherwise harmful
    return l_diff




