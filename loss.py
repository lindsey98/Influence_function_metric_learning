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
from hard_sample_detection.gmm import *

def masked_softmax(A, dim, t=1.0):
    '''
        Apply softmax but ignore zeros
        :param A: torch.tensor of shape (N, C)
        :param t: temperature
        :return A_softmax: masked softmax of A
    '''
    A_exp = torch.exp(A)
    A_exp = A_exp * (A != 0).float()  # this step masks zero entries
    temp_A = (A_exp)**(1/t)
    A_softmax = temp_A / torch.sum(temp_A, dim=dim, keepdim=True)
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
    def __init__(self, nb_classes: int, sz_embed: int, scale: float, initial_proxy_num: int, tau:float=0.0, **kwargs):
        '''
            :param nb_classes: number of classes in training set
            :param sz_embed: embedding size, e.g. 2048, 512, 64
            :param scale: self.scale is equal to sqrt(1/Temperature), default is 3
        '''

        torch.nn.Module.__init__(self)
        self.max_proxy_per_class = 5  # maximum number of proxies per class
        self.current_proxy = [initial_proxy_num] * nb_classes  # start with single proxy per class
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes * self.max_proxy_per_class, sz_embed) / 8)
        self.mask = torch.zeros((nb_classes, self.max_proxy_per_class), requires_grad=False).to('cuda')
        self.scale = scale  # sqrt(1/Temperature)
        self.nb_classes = nb_classes  # number of classes
        self.tau = tau

        self.create_mask()  # create initial mask

    def create_mask(self):
        '''
            initialize mask on proxies, only first proxy for each class is activated
        '''
        for c in range(self.nb_classes):
            proxy4cls = self.current_proxy[c]
            self.mask[c, 0:proxy4cls] = 1

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
            self.mask[cls, :self.current_proxy[cls]] = 1  # unfreeze mask
        return

    def regularization(self, Proxy_IP):
        A = []
        for x in range(self.nb_classes):
            count_proxy = int(torch.sum(self.mask[x]).item())
            A_this = torch.zeros((self.max_proxy_per_class, self.max_proxy_per_class))
            A_this[:count_proxy, :count_proxy] = 1.
            A.append(A_this)
        block_mask = torch.block_diag(*A).to(Proxy_IP.device)
        masked_proxyIP = torch.mul(block_mask, Proxy_IP)
        regularization = torch.sum(masked_proxyIP)
        nonzero_items = torch.count_nonzero(masked_proxyIP).item()
        if nonzero_items == 0:
            mean_regularization = regularization * 0.0
        else:
            mean_regularization = regularization / (nonzero_items) # divide by 2 because proxy inner product was

        return mean_regularization

    @torch.no_grad()
    def loss4debug(self, X:torch.Tensor, indices: torch.Tensor, T:torch.Tensor):
        '''
            Return intermediate calculations of loss for debugging purpose
        '''
        P = self.proxies
        temperature = self.scale
        P = F.normalize(P, p=2, dim=-1) * temperature
        X = F.normalize(X, p=2, dim=-1) * temperature
        P = P.to(X.device)

        # pairwise distance
        D, IP = pairwise_distance(
            torch.cat([X, P]),
            squared=True
        ) # of shape (N, C*maxP)
        D = D[:X.size()[0], X.size()[0]:] # of shape (N, N)
        Proxy_IP = IP[X.size()[0]:, X.size()[0]:] # of shape (C*maxP, C*maxP)
        IP = IP[:X.size()[0], X.size()[0]:] # of shape (N, C*maxP)

        self.mask = self.mask.to(X.device)
        D_reshape = D.reshape((X.shape[0], self.nb_classes, self.max_proxy_per_class))  # of shape (N, C, maxP)
        output = D_reshape * self.mask.unsqueeze(0)  # mask unactivated proxies
        output_IP = IP.reshape((X.shape[0], self.nb_classes, self.max_proxy_per_class)) * self.mask.unsqueeze(0)
        normalize_prob = masked_softmax(output_IP, t=0.1) # p(i,c)
        D_weighted = torch.sum(normalize_prob * output, dim=-1)  # S_i,c: weighted sum of distance, reduce to shape (N, C)

        smoothing_const = 0.0 # smoothing class labels
        target_probs = (torch.ones((X.shape[0], self.nb_classes)) * smoothing_const).to(T.device)
        target_probs.scatter_(1, T.unsqueeze(1), 1 - smoothing_const) # one-hot label

        gt_prob = normalize_prob[torch.arange(X.size()[0]), T.long(), :]
        gt_D_weighted = D_weighted[torch.arange(X.size()[0]), T.long()]

        base_loss = torch.sum(- target_probs * F.log_softmax(-D_weighted, -1), -1)

        return indices, gt_prob, gt_D_weighted, base_loss, Proxy_IP


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

        # pairwise distance
        D, IP = pairwise_distance(
            torch.cat([X, P]),
            squared=True
        ) # of shape (N, C*maxP)
        D = D[:X.size()[0], X.size()[0]:]
        Proxy_IP = IP[X.size()[0]:, X.size()[0]:] # between-proxy similarity
        Proxy_IP = ((Proxy_IP + 1.)/2) * (1 - torch.eye(Proxy_IP.shape[0], Proxy_IP.shape[0]).to(Proxy_IP.device)) # add 1 to avoid zero entries
        IP = IP[:X.size()[0], X.size()[0]:]

        D_reshape = D.reshape((X.shape[0], self.nb_classes, self.max_proxy_per_class))  # of shape (N, C, maxP)
        output = D_reshape * self.mask.unsqueeze(0)  # mask unactivated proxies
        # normalize_prob_orig = masked_softmax(-output, t=1) # low distance proxy should get higher weight
        #FIXME: here use inner product to compute intra-class weight
        output_IP = IP.reshape((X.shape[0], self.nb_classes, self.max_proxy_per_class)) * self.mask.unsqueeze(0)
        normalize_prob = masked_softmax(output_IP, t=0.1)

        D_weighted = torch.sum(normalize_prob * output, dim=-1)  # weighted sum of distance, reduce to shape (N, C)

        smoothing_const = 0.0 # smoothing class labels
        target_probs = (torch.ones((X.shape[0], self.nb_classes)) * smoothing_const).to(T.device)
        target_probs.scatter_(1, T.unsqueeze(1), 1 - smoothing_const) # one-hot label

        base_loss = torch.sum(- target_probs * F.log_softmax(-D_weighted+(1e-12), -1), -1).mean() # log underflow
        if self.tau == 0.0:
            return base_loss
        else:
            mean_regularization = self.regularization(Proxy_IP)
            loss = base_loss + mean_regularization * self.tau

            return loss


class ProxyNCA_prob_orig(torch.nn.Module):
    '''
        Original loss in ProxyNCA++
    '''
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, X, indices, T):
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
        )[0][:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss


class ProxyNCA_prob_kd(torch.nn.Module):
    '''
        Original loss in ProxyNCA++
    '''
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, X, indices, T):
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
        )[0][:X.size()[0], X.size()[0]:] # of shape (N,C)
        D_sim = F.softmax(-D, -1) # of shape (N,C)
        logD_sim = torch.log(D_sim) # of shape (N,C)

        # compute between-instance CE
        label_equal_mask = (T == T.unsqueeze(0).T) * (1-torch.eye(X.size()[0]).to(T.device)) # of shape (N, N)
        innerCE = torch.einsum('ij,bj->ibj', D_sim, logD_sim) # of shape (N, N, C)
        innerCE_loss = torch.sum(-innerCE, -1) # of shape (N, N)
        innerCE_loss = label_equal_mask * innerCE_loss # mask different class pairs # of shape (N, N)
        innerCE_loss = innerCE_loss[innerCE_loss.nonzero(as_tuple=True)] # take non-zero
        innerCE_loss = torch.mean(innerCE_loss) # take mean

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )

        loss = torch.sum(- T * logD_sim, -1) # -ylog(yhat)
        loss = loss.mean() + 0.2*innerCE_loss
        return loss


class ProxyNCA_prob_mixup(torch.nn.Module):
    '''
        Uncertainty-guided MixUp
    '''

    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale
        self.sz_embed = sz_embed
        self.nb_classes = nb_classes

    def forward(self, X, indices, T):
        P = self.proxies
        # note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2

        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        _, IP = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )
        IP = IP[: X.size()[0], X.size()[0]:]

        # T_sub, virtual_X = self.informative_mixup(X=X, T=T, IP=IP, P=P)
        # Xall = torch.cat((X, self.scale * virtual_X), dim=0)
        # Tall = torch.cat([T, T_sub])
        # Dall = pairwise_distance(
        #     torch.cat(
        #         [Xall, P]
        #     ),
        #     squared=True
        # )[0][: Xall.size()[0], Xall.size()[0]:]
        #
        # Tall = binarize_and_smooth_labels(
        #     T=Tall, nb_classes=len(P), smoothing_const=0
        # )
        # loss = torch.sum(- Tall * F.log_softmax(-Dall, -1), -1)
        # loss = loss.mean()

        virtual_classes, virtual_samples = self.informative_mixup(X=X, T=T, IP=IP, P=P)
        Xall = torch.cat((X, self.scale * virtual_samples), dim=0) # (N+|V|, sz_embed)
        T = binarize_and_smooth_labels(
           T=T, nb_classes=self.nb_classes, smoothing_const=0 # FIXME
        ) # (N, C)
        Tall = torch.cat([T, virtual_classes], 0) # (N+|V|, C)

        Dall = pairwise_distance(
                torch.cat(
                    [Xall, P]
                ),
                squared=True
        )[0][: Xall.size()[0], Xall.size()[0]:] # (N+|V|, C)

        loss = torch.sum(- Tall * F.log_softmax(-Dall, -1), -1)
        loss = loss.mean()

        return loss

    @staticmethod
    def random_lambdas(n_samples, alpha=1.0):
        lambdas = []
        for _ in range(n_samples):
            lambdas.append(np.random.beta(alpha, alpha))
        return torch.tensor(lambdas)

    @staticmethod
    def uncertainty_lambdas(index1s, index2s, T, IP):
        C1, C2 = T[index1s], T[index2s]
        X1P1, X1P2 = torch.clamp(IP[index1s, C1], min=-1., max=1.), torch.clamp(IP[index1s, C2], min=-1., max=1.) # (n_samples, )
        X2P1, X2P2 = torch.clamp(IP[index2s, C1], min=-1., max=1.), torch.clamp(IP[index2s, C2], min=-1., max=1.)
        lambdas = (X2P2-X2P1)/((X2P2-X2P1) + (X1P1-X1P2))
        lambdas = torch.clamp(lambdas, min=0.3, max=0.7)

        return lambdas

    def intracls_mixup(self, X, T, IP, pairs_per_cls, method=1):

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(self.proxies), smoothing_const=0
        )
        D2T = IP * T  # (N, C) sparse matrix
        non_empty_mask = D2T.abs().sum(dim=0).bool()
        T_sub = torch.where(non_empty_mask == True)[0]  # record the class labels
        if pairs_per_cls > 1:
            T_sub = torch.repeat_interleave(T_sub, pairs_per_cls)

        D2T_normalize = D2T[:, non_empty_mask]  # (N, C_sub) filter out non-zero columns which are the classes not sampled in this batch
        D2T_normalize = masked_softmax(D2T_normalize, dim=0)  # normalize for each class to find highest confidence samples to mix (N, C_sub)

        # weighted sampling weighted by confidence
        if method == 1:
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(pairs_per_cls):
                    pair_ind = np.random.choice(D2T_normalize.size()[0], 2,
                                                p=D2T_normalize[:, j].detach().cpu().numpy(),
                                                replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long() # (2, C_sub*pairs_percls)

        elif method == 2:
            # purely random sampling
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(pairs_per_cls):
                    pair_ind = np.random.choice(D2T_normalize.size()[0], 2,
                                                replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long() # (2, C_sub*pairs_percls)

        else:
            raise NotImplementedError

        selectedX_samecls = torch.cat([X[index, :].unsqueeze(-1) for index in mixup_ind_samecls], dim=-1)  # of shape (C_sub*pairs_percls, sz_embed, 2)
        # sample lambda coefficients (diff lambda for diff samples)
        lambda_samecls = self.random_lambdas(selectedX_samecls.size()[0])
        lambda_samecls = lambda_samecls.unsqueeze(-1).repeat(1, self.sz_embed)
        lambda_samecls = torch.stack((lambda_samecls, 1.-lambda_samecls), dim=-1).to(selectedX_samecls.device)  # (C_sub*pairs_percls, sz_embed, 2)

        # mixup
        virtual_samples = torch.sum(selectedX_samecls * lambda_samecls, dim=-1)  # (C_sub*pairs_percls, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)
        assert T_sub.size()[0] == virtual_samples.size()[0]
        return T_sub, virtual_samples

    def intercls_mixup(self, X, T, P, IP, method=1):
        # we only sythesize samples and interpolating class labels
        # get all inter-class pairs to mixup
        index1s, index2s = torch.tensor([]), torch.tensor([])
        for index1 in range(X.size()[0]):
            for index2 in range(index1, X.size()[0]):
                if index1 == index2 or T[index1] == T[index2]:
                    continue
                index1s = torch.cat((index1s, torch.tensor([index1])))
                index2s = torch.cat((index2s, torch.tensor([index2])))

        index1s, index2s = index1s.long(), index2s.long()
        cls_index1s, cls_index2s = T[index1s], T[index2s]

        if method == 1:
            # uncertainty-based sampling
            lambdas_diffcls = self.uncertainty_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)
        elif method == 2:
            # pure random sampling
            lambdas_diffcls = self.random_lambdas(len(index1s)).unsqueeze(-1).repeat(1, self.sz_embed)
        else:
            raise NotImplementedError

        selectedX_diffcls = torch.stack([X[index1s, :], X[index2s, :]], dim=-1) # of shape (n_samples, sz_embed, 2)
        lambdas_diffcls = torch.stack((lambdas_diffcls, 1.-lambdas_diffcls), dim=-1).to(selectedX_diffcls.device) # (n_samples, sz_embed, 2)
        virtual_samples = torch.sum(selectedX_diffcls * lambdas_diffcls, dim=-1) # (n_samples, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)

        # virtual class
        C1 = binarize_and_smooth_labels( T=cls_index1s, nb_classes=self.nb_classes, smoothing_const=0 ) # (n_samples, C)
        C2 = binarize_and_smooth_labels( T=cls_index2s, nb_classes=self.nb_classes, smoothing_const=0 ) # (n_samples, C)
        C = torch.stack((C1, C2), dim=-1) # (n_samples, C, 2)
        lambdas_diffcls = lambdas_diffcls[:, :self.nb_classes, :] # same lambdas used as when creating virtual proxies
        virtual_classes = torch.sum(C * lambdas_diffcls, dim=-1) # (n_samples, C)

        assert virtual_classes.size()[0] == virtual_samples.size()[0]
        return virtual_classes, virtual_samples

    def intercls_mixup_proxy(self, X, T, P, IP, method=2):
        # We synthesize proxies as well as samples, where the new proxies are treated as new classes
        # get all inter-class pairs to mixup
        index1s, index2s = torch.tensor([]), torch.tensor([])
        for index1 in range(X.size()[0]):
            for index2 in range(index1, X.size()[0]):
                if index1 == index2 or T[index1] == T[index2]:
                    continue
                index1s = torch.cat((index1s, torch.tensor([index1])))
                index2s = torch.cat((index2s, torch.tensor([index2])))

        index1s, index2s = index1s.long(), index2s.long()
        cls_index1s, cls_index2s = T[index1s], T[index2s]

        # virtual proxies and virtual class labels
        P_pairs = torch.stack((P[cls_index1s, :], P[cls_index2s, :]), dim=0)  # (2, n_samples, sz_embed)
        lambdas = 0.5 * torch.ones((1, P_pairs.size()[1], P_pairs.size()[2]))  # universal lambdas = 0.5
        lambdas = torch.cat((lambdas, 1. - lambdas), dim=0)  # (2, n_samples, sz_embed)
        virtual_proxies = torch.sum(P_pairs * lambdas, dim=0)  # (n_samples, sz_embed)
        virtual_proxies = F.normalize(virtual_proxies, p=2, dim=-1) # (n_samples, sz_embed)
        virtual_proxies_reduced, virtual_classes = torch.unique(virtual_proxies, dim=0, return_inverse=True, sorted=False)
        virtual_classes = self.nb_classes + virtual_classes # new class labels

        # virtual samples
        if method == 1:
            lambdas_diffcls = self.uncertainty_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)
        elif method == 2:
            lambdas_diffcls = self.random_lambdas(len(index1s)).unsqueeze(-1).repeat(1, self.sz_embed)
        else:
            raise NotImplementedError

        selectedX_diffcls = torch.stack([X[index1s, :], X[index2s, :]], dim=-1)  # of shape (n_samples, sz_embed, 2)
        lambdas_diffcls = torch.stack((lambdas_diffcls, 1.-lambdas_diffcls), dim=-1).to(selectedX_diffcls.device)  # (n_samples, sz_embed, 2)
        virtual_samples = torch.sum(selectedX_diffcls * lambdas_diffcls, dim=-1)  # (n_samples, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)

        assert virtual_classes.size()[0] == virtual_samples.size()[0]
        return virtual_proxies_reduced, virtual_samples, virtual_classes

    def informative_mixup(self, X, T, IP, P):
        '''
            Perform informative MixUp
                :param X:
                :param IP:
                :param T: (N, )
                :param mixup_ratio:
        '''

        # between different class (keep uncertain only)
        virtual_classes, virtual_samples = self.intercls_mixup(X, T, P, IP)
        return virtual_classes, virtual_samples

        # between same class (high confidence only)
        # T_sub, virtual_samples = self.intracls_mixup(X, T, IP, pairs_per_cls=4)
        # return T_sub, virtual_samples

class ProxyNCA_distribution_loss(torch.nn.Module):
    '''
        ProxyNCA distribution based loss
    '''
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        # self.sigmas_inv = torch.nn.Parameter(torch.ones(nb_classes, sz_embed))
        self.sigmas_inv = torch.nn.Parameter(torch.ones(nb_classes, sz_embed)*math.log(math.e-1)) # if softplus activation is used
        # self.sigmas_inv.requires_grad = False # FIXME: initial test

        self.sz_embed = sz_embed
        self.nb_classes = nb_classes
        self.scale = scale

    def center_init(self, AllX, AllY):
        initial_centers = torch.tensor([])
        for cls in range(self.nb_classes):
            selectedX = AllX[AllY == cls]
            if len(selectedX) == 0:
                selectedX_mean = torch.randn(self.sz_embed)
                initial_centers = torch.cat((initial_centers, selectedX_mean.unsqueeze(0)), dim=0)
            else:
                selectedX_mean = selectedX.mean(0)
                # print(selectedX_mean.shape)
                initial_centers = torch.cat((initial_centers, selectedX_mean.unsqueeze(0)), dim=0)
        self.proxies.data = initial_centers.data

    def kmeans_init(self, AllX):
        initial_centers, _ = kmeans_fun_gpu(AllX, K=self.nb_classes)
        self.proxies.data = initial_centers.data

    def kl_divergence(self, trace_Sigma):
        Sigmas = 1 / trace_Sigma # (C, sz_embed)
        mus = F.normalize(self.proxies, p=2, dim=-1) # (C, sz_embed)
        mus2 = torch.square(mus)
        KL = 0.5*(Sigmas + mus2 - 1 - torch.log(Sigmas)) # (C, sz_embed)
        KL = KL.sum(-1).mean()
        return KL

    def forward(self, X, indices, T):
        P = self.proxies
        Sigma_inv = torch.diag_embed(F.softplus(self.sigmas_inv)**2) # of shape (C, sz_embed, sz_embed)

        # mixupX = self.informative_mixup(X, P)

        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1) # (N, sz_embed)

        trace_Sigma_inv = Sigma_inv[:, torch.arange(self.sz_embed), torch.arange(self.sz_embed)] # (C, sz_embed)
        xPdist = X.unsqueeze(1) - P # (N, C, sz_embed)
        xPdist2 = xPdist * xPdist # Hadamard product (N, C, sz_embed)

        xSx = torch.matmul(xPdist2, trace_Sigma_inv.T) # (N, C, C)
        xSx = xSx[:, torch.arange(self.nb_classes), torch.arange(self.nb_classes)] # (N, C)
        D = xSx # compute the -(x-mu)'Sigma^-1(x-mu) of shape (N, C)

        D_sim = F.softmax(-D, -1) # similarity to proxies of shape (N, C)
        logD_sim = torch.log(D_sim) # of shape (N, C)

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )

        loss = torch.sum(- T * logD_sim, -1) # -ylog(yhat)
        loss = loss.mean()
        kl_loss = self.kl_divergence(trace_Sigma_inv)
        loss = loss + 0.2*kl_loss # KL regularization on prior N(0, I)

        return loss




