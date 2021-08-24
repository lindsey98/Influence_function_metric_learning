from similarity import pairwise_distance
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy
import math
from dataset.base import SubSampler
from torch.utils.data import DataLoader

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T

class ProxyNCA_classic(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, X, T):
        P = self.proxies

        #note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
        P = self.scale * F.normalize(P, p = 2, dim = -1)
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:] # only get distance between X and P, not self-distance within X or within P

        T = binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = 0
        ) # smoothing constant =0 means no smoothing, just one-hot label
        loss1 = torch.sum(T * torch.exp(-D), -1)
        loss2 = torch.sum((1-T) * torch.exp(-D), -1)
        loss = -torch.log(loss1 / loss2)
        loss = loss.mean()
        return loss

class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, len_training, **kwargs):
        torch.nn.Module.__init__(self)
        self.max_proxy_per_class = 5 # maximum number of proxies per class
        self.current_proxy = [1]*nb_classes # start with single proxy per class
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes*self.max_proxy_per_class,  sz_embed) / 8)
        self.mask = torch.zeros((nb_classes, self.max_proxy_per_class), requires_grad=False).to('cuda')
        self.scale = scale # temperature
        self.len_training = len_training # training
        self.cached_sim = np.zeros(len_training) # cache the similarity to ground-truth proxy
        self.cached_cls = np.zeros(len_training)
        self.nb_classes = nb_classes # number of classes

        self.create_mask()  # create initial mask

    def create_mask(self):
        # create mask on proxies
        for c in range(self.nb_classes):
            proxy4cls = self.current_proxy[c]
            self.mask[c, :proxy4cls] = 1

    def reinitialize_cache_sim(self):
        # initialze cached inner product list
        self.cached_sim = np.zeros(self.len_training)

    def add_proxy(self, cls, new_proxy):
        cls = int(cls)
        if self.current_proxy[cls] == self.max_proxy_per_class:
            pass
        else:
            self.proxies.data[self.max_proxy_per_class * cls + self.current_proxy[cls], :] = new_proxy.data  # initilaize new proxy there
            self.current_proxy[cls] += 1 # update number of proxy for this class
            self.mask[cls, self.current_proxy[cls]] = 1 # unfreeze mask
        return

    @torch.no_grad()
    def inner_product_sim(self, X, P, T):
        # get maximum inner product to ground-truth proxy
        input = F.normalize(X, dim=-1, p=2)
        mask = self.mask.view(self.nb_classes * self.max_proxy_per_class, -1).to(P.device)
        masked_P = F.normalize(P, dim=-1, p=2) * mask
        IP = torch.mm(input, masked_P.T)  # inner product between X and P of shape (N, maxP*C)
        IP_reshape = torch.reshape(IP, (X.size(0), self.nb_classes, self.max_proxy_per_class)) # reshape inner product as shape of (N, C, maxP)
        cls_labels = T.nonzero()[:, 1]
        IP_gt = IP_reshape[torch.arange(len(X)), cls_labels.long(), :] # of shape (N, maxP)
        L_IP, _ = torch.max(IP_gt, dim=-1) # of shape (N,)

        return L_IP, cls_labels


    def forward(self, X, indices, T):
        P = self.proxies
        #note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
       
        P = self.scale * F.normalize(P, p = 2, dim = -1)
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        D = pairwise_distance(
            torch.cat([X, P]),
            squared = True
        )[:X.size()[0], X.size()[0]:] # of shape (N, maxP*C)

        # FIXME: no gradient to X, but has gradient to self.proxies
        D_reshape = D.reshape((X.size()[0], self.nb_classes, self.max_proxy_per_class)) # of shape (N, C, maxP)
        prob = F.softmax(self.mask * (-D_reshape), dim=-1)
        D_weighted = torch.sum(prob * (self.mask * D_reshape), dim=-1) # # low distance proxy get higher weights weighted distance, reduce to shape (N, C)

        T = binarize_and_smooth_labels(
            T = T, nb_classes = self.nb_classes, smoothing_const = 0 # one-hot gt label
        ) # smooth one-hot label

        # TODO: multiple proxies per class
        loss = torch.sum(- T * F.log_softmax(-D_weighted, -1), -1)
        # print(loss)

        if indices is not None:
            L_IP, cls_labels = self.inner_product_sim(X, P, T)
            self.cached_sim[indices] = L_IP.detach().cpu().numpy() # cache losses for each training sample
            self.cached_cls[indices] = cls_labels.detach().cpu().numpy()
        return loss.mean()




