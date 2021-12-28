#! /usr/bin/env python3
import torch
from torch.autograd import grad
from tqdm import tqdm
import numpy as np

def calc_loss_train(model, dl, criterion, indices=None):
    l = []
    if indices is not None:
        for ct, (x, t, ind) in tqdm(enumerate(dl)):
            if ind.item() in indices:
                x, t = x.cuda(), t.cuda()
                m = model(x)
                l_this = criterion(m, None, t)
                l.append(l_this.detach().cpu().item())
    else:
        for ct, (x, t, _) in tqdm(enumerate(dl)):
            x, t = x.cuda(), t.cuda()
            m = model(x)
            l_this = criterion(m, None, t)
            l.append(l_this.detach().cpu().item())
    return l

def loss_change_train(model, criterion, dl_tr, params_prev, params_cur):

    weight_orig = model.module[-1].weight.data # cache original parameters
    model.module[-1].weight.data = params_prev
    l_prev = calc_loss_train(model, dl_tr, criterion, None)

    model.module[-1].weight.data = params_cur
    l_cur = calc_loss_train(model, dl_tr, criterion, None)

    model.module[-1].weight.data = weight_orig # dont forget to revise the weights back to the original
    return np.asarray(l_prev), np.asarray(l_cur)

def calc_inter_dist_pair(feat_cls1, feat_cls2):
    inter_dis = torch.cdist(feat_cls1, feat_cls2).square()  # inter class distance
    inter_dis = inter_dis.diagonal().sum() # aligned pairs
    return inter_dis

def grad_confusion(model, all_features, cls, confusion_classes,
                   pred, label, nn_indices):

    pred = pred.detach().cpu().numpy(); label = label.detach().cpu().numpy()
    nn_indices = nn_indices.flatten()

    # Get cls samples indices and confusion_classes samples indices
    interest_indices = [[] for _ in range(len(confusion_classes))] # belong to cls and wrongly predicted
    closest_cls_indices = [[] for _ in range(len(confusion_classes))] # belong to confusion classes and are neighbors of interest_indices
    pair_counts = 0 # count how many confusion pairs
    for kk, confusion_cls in enumerate(confusion_classes):
        wrong_as_confusion_cls_indices = np.where((pred == confusion_cls) & (label == cls))[0]
        interest_indices[kk] = wrong_as_confusion_cls_indices
        closest_cls_indices[kk] = nn_indices[wrong_as_confusion_cls_indices]
        pair_counts += len(wrong_as_confusion_cls_indices)

    # Compute pairwise confusion and record gradients to projection layer's weights
    accum_grads = [torch.zeros_like(model.module[-1].weight).detach().cpu()]
    accum_confusion = 0.
    for kk in range(len(confusion_classes)):
        cls_features = all_features[interest_indices[kk]]
        confuse_cls_features = all_features[closest_cls_indices[kk]]
        cls_features = cls_features.cuda()
        confuse_cls_features = confuse_cls_features.cuda()

        feature1 = model.module[-1](cls_features)
        feature2 = model.module[-1](confuse_cls_features)
        confusion = calc_inter_dist_pair(feature1, feature2)

        params = model.module[-1].weight
        grad_confusion2params = list(grad(confusion, params))
        accum_grads = [x + y.detach().cpu() for x, y in zip(accum_grads, grad_confusion2params)] # accumulate gradients
        accum_confusion += confusion.detach().cpu().item() # accumulate confusion

    accum_grads = [x/pair_counts for x in accum_grads]
    accum_confusion = accum_confusion/pair_counts
    return accum_confusion, accum_grads

def calc_influential_func_sample(grad_alltrain):
    l_prev = grad_alltrain['l_prev']
    l_cur = grad_alltrain['l_cur']
    l_diff = np.stack(l_cur) - np.stack(l_prev) # l'-l0, if l_diff < 0, helpful
    return l_diff

# def calc_inter_dist(feat_cls1, feat_cls2):
#     '''
#         Calculate inter class distance
#         Arguments:
#             feat_cls1
#             feat_cls2
#         Returns:
#             inter dist
#     '''
#     n1, n2 = feat_cls1.size()[0], feat_cls2.size()[0]
#     inter_dis = torch.cdist(feat_cls1, feat_cls2).square()  # inter class distance
#     inter_dis = inter_dis.sum() / (n1*n2)
#     return inter_dis
# def calc_intravar(feat):
#     '''
#         Get intra-class variance (unbiased estimate)
#     '''
#     n = feat.size()[0]
#     intra_var = ((feat - feat.mean(0)) ** 2).sum() / n
#     return intra_var
# def grad_interdist(model, dl_ev, cls1, cls2, limit=50):
#     '''
#         Calculate class confusion or gradient of class confusion to model parameters
#         Arguments:
#             model: torch NN, model used to evaluate the dataset
#             dl_ev: test dataloader
#             cls1: class 1
#             cls2: class 2
#             get_grad: compute gradient or the original
#         Returns:
#             grad_confusion2params: gradient of confusion to params
#     '''
#     feat_cls1 = torch.tensor([]).cuda()  # (N1, sz_embed)
#     feat_cls2 = torch.tensor([]).cuda()  # (N2, sz_embed)
#     model.eval(); model.zero_grad()
#     for ct, (x, t, _) in tqdm(enumerate(dl_ev)): # need to recalculate the feature embeddings since we need the gradient
#         if len(feat_cls1) >= limit and len(feat_cls2) >= limit:
#             break
#         if t.item() == cls1: # belong to class 1
#             if len(feat_cls1) >= limit:
#                 continue
#             x = x.cuda()
#             m = model(x)
#             feat_cls1 = torch.cat((feat_cls1, m), dim=0)
#         elif t.item() == cls2: # belong to class 2
#             if len(feat_cls2) >= limit:
#                 continue
#             x = x.cuda()
#             m = model(x)
#             feat_cls2 = torch.cat((feat_cls2, m), dim=0)
#         else: # skip irrelevant test samples
#             pass
#
#     confusion = calc_inter_dist(feat_cls1, feat_cls2)
#     # confusion = calc_inter_dist_pair(feat_cls1, feat_cls2)
#     params = model.module[-1].weight # last linear layer
#     grad_confusion2params = list(grad(confusion, params))
#     return confusion, grad_confusion2params


# def grad_intravar(model, dl_ev, cls):
#     '''
#         Calculate class confusion or gradient of class confusion to model parameters
#         Arguments:
#             model: torch NN, model used to evaluate the dataset
#             dl_ev: test dataloader
#             cls1: class 1
#             cls2: class 2
#             get_grad: compute gradient or the original
#         Returns:
#             grad_confusion2params: gradient of confusion to params
#     '''
#     feat_cls = torch.tensor([]).cuda()  # (N, sz_embed)
#     model.eval(); model.zero_grad()
#     for ct, (x, t, _) in tqdm(enumerate(dl_ev)): # need to recalculate the feature embeddings since we need the gradient
#         if t.item() == cls: # belong to class
#             x = x.cuda()
#             m = model(x)
#             feat_cls = torch.cat((feat_cls, m), dim=0)
#         else: # skip irrelevant test samples
#             pass
#
#     intra_var = calc_intravar(feat_cls)
#     params = model.module[-1].weight # last linear layer
#     grad_intravar2params = list(grad(intra_var, params)) # d(var)/d(theta)
#     return intra_var, grad_intravar2params


