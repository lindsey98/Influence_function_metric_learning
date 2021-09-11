from similarity import pairwise_distance
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy
import math
from dataset.base import SubSampler
from torch.utils.data import DataLoader
from typing import Union, List
import sklearn.preprocessing

def masked_softmax(A):
    '''
        Apply softmax but ignore zeros
        :param A: torch.tensor of shape (N, C)
        :return A_softmax: masked softmax of A
    '''
    A_exp = torch.exp(A)
    A_exp = A_exp * (A != 0).float()  # this step masks zero entries
    A_softmax = A_exp / torch.sum(A_exp, dim=-1, keepdim=True)
    return A_softmax

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0):
    '''
        Create smoother gt class labels
        :param T: torch.Tensor of shape (N,), gt class labels
        :param nb_classes: number of classes
        :param smoothing_const: smoothing factor, when =0, no smoothing is applied, just one-hot label
        :return T: torch.Tensor of shape (N, C)
    '''
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T


class ProxyNCA_classic(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        '''
            :param nb_classes: number of classes in training set
            :param sz_embed: embedding size, e.g. 2048, 512, 64
            :param scale: self.scale is equal to sqrt(1/Temperature), default is 3
        '''
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, X, T):
        '''
            Forward propogation of loss
            :param X: the embedding torch.Tensor of shape (N, sz_embed)
            :param T: a torch.Tensor of shape (N, ), keeping class labels: sample -> class label
            :return : batch average loss
        '''
        P = self.proxies

        # note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[:X.size()[0], X.size()[0]:]  # only get distance between X and P, not self-distance within X or within P

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )  # smoothing constant =0 means no smoothing, just one-hot label
        loss1 = torch.sum(T * torch.exp(-D), -1)
        loss2 = torch.sum((1 - T) * torch.exp(-D), -1) # here the denominator is the sum over all other classes
        loss = -torch.log(loss1 / loss2) # There is no guarantee that k = (exp(-positive_distance)/ exp(-negative_distance)) is below 1.
                                         # if k is greater than one, algorithm will give you negative loss values. (-log(k))
        loss = loss.mean()
        return loss


class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, nb_classes: int, sz_embed: int, scale: float, len_training: int, **kwargs):
        '''
            :param nb_classes: number of classes in training set
            :param sz_embed: embedding size, e.g. 2048, 512, 64
            :param scale: self.scale is equal to sqrt(1/Temperature), default is 3
            :param len_training: number of training examples in train loader
        '''

        torch.nn.Module.__init__(self)
        self.max_proxy_per_class = 5  # maximum number of proxies per class
        self.current_proxy = [1] * nb_classes  # start with single proxy per class
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes * self.max_proxy_per_class, sz_embed) / 8)
        self.mask = torch.zeros((nb_classes, self.max_proxy_per_class), requires_grad=False).to('cuda')
        self.scale = scale  # sqrt(1/Temperature)
        self.len_training = len_training  # training
        self.cached_sim = np.zeros(len_training)  # cache the similarity to ground-truth proxy
        self.cached_cls = np.zeros(len_training) # cache the gt-class
        self.nb_classes = nb_classes  # number of classes

        self.create_mask()  # create initial mask

    def create_mask(self):
        '''
            initialize mask on proxies, only first proxy for each class is activated
        '''
        for c in range(self.nb_classes):
            proxy4cls = self.current_proxy[c]
            self.mask[c, 0:proxy4cls] = 1

    def reinitialize_cache_sim(self):
        '''
            initialze cached inner product similarity list to be all zeros
        '''
        self.cached_sim = np.zeros(self.len_training)

    def add_proxy(self, cls: Union[int, float], new_proxy: torch.Tensor):
        '''
            activate one more proxy for class cls
            :param cls: the class to be added a proxy
            :param new_proxy: the added proxy (mean of hard examples)
        '''
        cls = int(cls)
        if self.current_proxy[cls] == self.max_proxy_per_class: # if reach the maximum proxy one class can get
            pass
        else:
            self.proxies.data[self.max_proxy_per_class * cls + self.current_proxy[cls], :] = new_proxy.data  # initilaize new proxy there
            self.current_proxy[cls] += 1  # update number of proxy for this class
            self.mask[cls, self.current_proxy[cls]] = 1  # unfreeze mask
        return

    @torch.no_grad()
    def inner_product_sim(self, X: torch.Tensor, P: torch.nn.Parameter, T: torch.Tensor):
        '''
            get maximum inner product similarity to ground-truth proxy
            :param X: embedding torch.Tensor of shape (N, sz_embed)
            :param P: learnt proxy torch.Tensor of shape (C * self.max_proxy_per_class, sz_embed)
            :param T: one-hot ground-truth class label torch.Tensor of shape (N, C)
            :return L_IP: Inner product similarity to the closest gt-class proxies
            :return cls_labels: gt-class labels
        '''
        X_copy = X.clone()
        P_copy = P.clone()
        T_copy = T.clone()

        X_copy = F.normalize(X_copy, dim=-1, p=2)
        P_copy = F.normalize(P_copy, dim=-1, p=2)
        mask = self.mask.view(self.nb_classes * self.max_proxy_per_class, -1).to(P_copy.device)
        masked_P = P_copy * mask # mask unactivated proxies
        IP = torch.mm(X_copy, masked_P.T)  # inner product between X and P of shape (N, C*maxP)
        IP_reshape = torch.reshape(IP, (X_copy.shape[0], self.nb_classes, self.max_proxy_per_class))  # reshape to (N, C, maxP)

        cls_labels = T_copy.nonzero()[:, 1] # get where the one-hot label is and that's the class label
        IP_gt = IP_reshape[torch.arange(X_copy.shape[0]), cls_labels.long(), :]  # get similarities to gt-class's proxies, of shape (N, maxP)
        L_IP, _ = torch.max(IP_gt, dim=-1)  # get maximum similarity to gt-class's proxies, of shape (N,)

        return L_IP, cls_labels

    def forward(self, X:torch.Tensor, indices: torch.Tensor, T:torch.Tensor):
        '''
            Forward propogation of loss
            :param X: the embedding torch.Tensor of shape (N, sz_embed)
            :param indices: a torch.Tensor of shape (N, ) of batch indices (used to track the loss/similarity back to a sample)
            :param T: a torch.Tensor of shape (N, ), keeping class labels: sample -> class label
            :return : batch average loss
        '''

        P = self.proxies
        temperature = self.scale
        P = F.normalize(P, p=2, dim=-1) * temperature
        X = F.normalize(X, p=2, dim=-1) * temperature

        # tensor
        D = pairwise_distance(
            torch.cat([X, P]),
            squared=True
        )[:X.size()[0], X.size()[0]:] # of shape (N, C*maxP)

        D_reshape = D.reshape((X.shape[0], self.nb_classes, self.max_proxy_per_class))  # of shape (N, C, maxP)
        output = D_reshape * self.mask.unsqueeze(0)  # mask unactivated proxies
        normalize_prob = masked_softmax(-output) # low distance proxy should get higher weight
        D_weighted = torch.sum(normalize_prob * output, dim=-1)  # weighted sum of distance, reduce to shape (N, C)

        smoothing_const = 0.0 # smoothing class labels
        target_probs = (torch.ones((X.shape[0], self.nb_classes)) * smoothing_const).to(T.device)
        target_probs.scatter_(1, T.unsqueeze(1), 1 - smoothing_const) # one-hot label

        loss = torch.sum(- target_probs * F.log_softmax(-D_weighted, -1), -1)

        if indices is not None:
            L_IP, cls_labels = self.inner_product_sim(X, P, target_probs)
            self.cached_sim[indices] = L_IP.detach().cpu().numpy()  # cache losses for each training sample
            self.cached_cls[indices] = cls_labels.detach().cpu().numpy()

        return loss.mean()


class ProxyNCA_prob_orig(torch.nn.Module):
    '''
        Original loss in ProxyNCA++
    '''
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, X, T):
        P = self.proxies
        # note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2

        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss
