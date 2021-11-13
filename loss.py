import numpy as np
import torch

from similarity import pairwise_distance
import torch.nn.functional as F
import math
from typing import Union
import sklearn.preprocessing
from itertools import combinations
import logging
from scipy.optimize import linear_sum_assignment
import utils


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

def smooth_labels(topk_ind, batch_size, nb_classes):
    '''
        Create smoother gt class labels
        :param T: torch.Tensor of shape (N,), gt class labels
        :param nb_classes: number of classes
        :param smoothing_const: smoothing factor, when =0, no smoothing is applied, just one-hot label
        :return T: torch.Tensor of shape (N, C)
    '''
    T = torch.zeros((batch_size, nb_classes))
    k = topk_ind.size()[1]
    for j in range(k):
        T[torch.arange(T.size()[0]), topk_ind[:, j]] = 1. / k
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

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed,
                 mixup_method='none', sampling_method=1,
                 shifts=4, pairs_per_cls=4,
                 mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        torch.nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.mixup_method = mixup_method
        self.sampling_method = sampling_method
        self.shifts = shifts
        self.pairs_per_cls = pairs_per_cls
        assert self.mixup_method in ['none']
        assert self.sampling_method in [1, 2]  # 1 is weighted sampling, 2 is random sampling

    @staticmethod
    def random_lambdas(alpha=1.0):
        return np.random.beta(alpha, alpha)  # sample a lambda from beta distribution

    @staticmethod
    def uncertainty_lambdas(index1s, index2s, T, IP):
        C1, C2 = T[index1s], T[index2s]
        X1P1, X1P2 = torch.clamp(IP[index1s, C1], min=-1., max=1.), \
                     torch.clamp(IP[index1s, C2], min=-1., max=1.)  # (n_samples, )
        X2P1, X2P2 = torch.clamp(IP[index2s, C1], min=-1., max=1.), \
                     torch.clamp(IP[index2s, C2], min=-1., max=1.)
        lambdas = (X2P2 - X2P1) / ((X2P2 - X2P1) + (X1P1 - X1P2))  # lambda_best
        noises = torch.from_numpy((np.random.uniform(size=len(index1s)) * 0.4 - 0.2)).to(lambdas.device)
        lambdas = lambdas + noises  # add small random noises ~ U(-0.2, 0.2)
        lambdas = torch.clamp(lambdas, min=0.2, max=0.8)  # clamp

        return lambdas

    def intracls_mixup(self, X, T, IP):

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(self.proxies), smoothing_const=0
        )  # (N, C)
        D2T = IP * T  # (N, C) sparse matrix
        non_empty_mask = D2T.abs().sum(dim=0).bool()
        T_sub = torch.where(non_empty_mask == True)[0]  # record the class labels
        T_sub = torch.repeat_interleave(T_sub, self.pairs_per_cls)

        D2T_normalize = D2T[:,
                        non_empty_mask]  # (N, C_sub) filter out non-zero columns which are the classes not sampled in this batch
        D2T_normalize = masked_softmax(D2T_normalize,
                                       dim=0)  # normalize for each class to find highest confidence samples to mix (N, C_sub)

        # weighted sampling weighted by inverted confidence
        if self.sampling_method == 1:
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(self.pairs_per_cls):
                    prob_vec = D2T_normalize[:, j].detach().cpu().numpy()  # take j column
                    index_pool = np.where(prob_vec != 0)[
                        0]  # find the indices withint the batch that takes class label j
                    prob_vec = (1. - prob_vec[index_pool])  # inverse its probability

                    if np.sum(prob_vec == 0) == len(prob_vec):  # by right, this should not happen
                        logging.info('All probabilities are zeros')
                        pair_ind = np.random.choice(index_pool, 2,
                                                    replace=False).tolist()
                    else:
                        pair_ind = np.random.choice(index_pool, 2,
                                                    p=(prob_vec + 1e-8) / (prob_vec.sum() + 1e-8),
                                                    # numerical stability
                                                    replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long()  # (2, C_sub*pairs_percls)

        # uniform random sampling
        elif self.sampling_method == 2:
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(self.pairs_per_cls):
                    pair_ind = np.random.choice(D2T_normalize.size()[0], 2,
                                                replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long()  # (2, C_sub*pairs_percls)

        selectedX_samecls = torch.stack([X[index, :] for index in mixup_ind_samecls],
                                        dim=-1)  # of shape (C_sub*pairs_percls, sz_embed, 2)

        # sample lambda coefficients
        lambda_samecls = self.random_lambdas() * torch.ones((len(selectedX_samecls), self.sz_embed))
        lambda_samecls = torch.stack((lambda_samecls, 1. - lambda_samecls), dim=-1).to(
            selectedX_samecls.device)  # (C_sub*pairs_percls, sz_embed, 2)

        # perform MixUp
        virtual_samples = torch.sum(selectedX_samecls * lambda_samecls, dim=-1)  # (C_sub*pairs_percls, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)
        assert T_sub.size()[0] == virtual_samples.size()[0]
        return T_sub, virtual_samples

    def intercls_mixup(self, X, T, IP):
        # we only sythesize samples and interpolating class labels
        # get some inter-class pairs to mixup
        index1s = torch.arange(X.size()[0])
        index2s = torch.roll(index1s,
                             -self.shifts)  # shifts should be equal to n_samples_per_cls in your balanced sampler
        index1s, index2s = index1s.long(), index2s.long()
        cls_index1s, cls_index2s = T[index1s], T[index2s]

        if self.sampling_method == 1:
            # uncertainty-based sampling
            lambdas_diffcls = self.uncertainty_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)
        elif self.sampling_method == 2:
            # pure random sampling
            lambdas_diffcls = self.random_lambdas() * torch.ones((len(index1s), self.sz_embed))

        # virtual samples
        selectedX_diffcls = torch.stack([X[index1s, :], X[index2s, :]], dim=-1)  # of shape (n_samples, sz_embed, 2)
        lambdas_diffcls = torch.stack((lambdas_diffcls, 1. - lambdas_diffcls), dim=-1).to(
            selectedX_diffcls.device)  # (n_samples, sz_embed, 2)
        virtual_samples = torch.sum(selectedX_diffcls * lambdas_diffcls, dim=-1)  # (n_samples, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)

        # interpolating class labels
        C1 = binarize_and_smooth_labels(T=cls_index1s, nb_classes=self.nb_classes, smoothing_const=0)  # (n_samples, C)
        C2 = binarize_and_smooth_labels(T=cls_index2s, nb_classes=self.nb_classes, smoothing_const=0)  # (n_samples, C)
        C = torch.stack((C1, C2), dim=-1)  # (n_samples, C, 2)
        lambdas_diffcls = lambdas_diffcls[:, :self.nb_classes, :]  # same lambdas used as when creating virtual proxies
        virtual_classes = torch.sum(C * lambdas_diffcls, dim=-1)  # (n_samples, C)

        assert virtual_classes.size()[0] == virtual_samples.size()[0]
        return virtual_classes, virtual_samples

    def forward(self, X, indices, T):
        P = self.proxies
        P = F.normalize(P, p=2, dim=-1)
        X = F.normalize(X, p=2, dim=-1)

        if self.mixup_method == 'none':
            # no MixUp applied
            cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
            P_one_hot = binarize_and_smooth_labels(T=T, nb_classes=self.nb_classes)
            N_one_hot = 1 - P_one_hot

            pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
            neg_exp = torch.exp(self.alpha * (cos + self.mrg))

            with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
            num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

            P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
            N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

            pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
            neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
            loss = pos_term + neg_term

        # if self.mixup_method == 'inter_noproxy':
        #     '''
        #         Original data
        #     '''
        #     cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        #     P_one_hot = binarize_and_smooth_labels(T=T, nb_classes=self.nb_classes)
        #     N_one_hot = 1 - P_one_hot
        #
        #     pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        #     neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        #
        #     with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
        #         dim=1)  # The set of positive proxies of data in the batch
        #     num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies
        #
        #     P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) # (C,)
        #     N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0) # (C,)
        #
        #     '''
        #         Synthetic data
        #     '''
        #     IP = pairwise_distance(torch.cat([X, P]),
        #         squared=True
        #     )[1][:X.size()[0], X.size()[0]:]  # (N, C)
        #
        #     # interclass without synthetic proxy
        #     virtual_classes, virtual_samples = self.intercls_mixup(X, T, IP)
        #     cos_smooth = F.linear(l2_norm(virtual_samples), l2_norm(P).double())  # Calcluate cosine similarity (N', C)
        #     P_smooth = virtual_classes # this is smooth label (N', C)
        #     N_smooth = 1 - P_smooth
        #
        #     pos_exp_smooth = torch.exp(-self.alpha * (cos_smooth - self.mrg)) # (N', C)
        #     neg_exp_smooth = torch.exp(self.alpha * (cos_smooth + self.mrg))
        #
        #     pos_exp_smooth = torch.where(P_smooth > 0, pos_exp_smooth * P_smooth, torch.zeros_like(pos_exp_smooth)).sum(dim=0) # (C,)
        #     neg_exp_smooth = torch.where(N_smooth == 1, neg_exp_smooth, torch.zeros_like(neg_exp_smooth)).sum(dim=0) # (C,)
        #
        #     pos_term = torch.log(1 + P_sim_sum + pos_exp_smooth).sum() / num_valid_proxies
        #     neg_term = torch.log(1 + N_sim_sum + neg_exp_smooth).sum() / self.nb_classes
        #     loss = pos_term + neg_term
        #
        # else:
        #     raise NotImplementedError

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

class ProxyNCA_prob_mixup(torch.nn.Module):
    '''
        Uncertainty-guided MixUp
    '''

    def __init__(self, nb_classes, sz_embed, scale, mixup_method, sampling_method, shifts, pairs_per_cls, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale
        self.sz_embed = sz_embed
        self.nb_classes = nb_classes
        self.mixup_method = mixup_method
        self.sampling_method = sampling_method
        self.shifts = shifts
        self.pairs_per_cls = pairs_per_cls
        assert self.mixup_method in ['none', 'intra', 'inter_noproxy', 'inter_proxy', 'both']
        assert self.sampling_method in [1, 2, 3] # 1 is weighted sampling, 2 is random sampling, 3 is reweighted sampling

    @staticmethod
    def random_lambdas(alpha=1.0):
        return np.random.beta(alpha, alpha) # sample a lambda from beta distribution

    @staticmethod
    def uncertainty_lambdas(index1s, index2s, T, IP):
        C1, C2 = T[index1s], T[index2s]
        X1P1, X1P2 = torch.clamp(IP[index1s, C1], min=-1., max=1.), \
                     torch.clamp(IP[index1s, C2], min=-1., max=1.) # (n_samples, )
        X2P1, X2P2 = torch.clamp(IP[index2s, C1], min=-1., max=1.), \
                     torch.clamp(IP[index2s, C2], min=-1., max=1.)
        lambdas = (X2P2-X2P1) / ((X2P2-X2P1) + (X1P1-X1P2)) # lambda_best
        noises = torch.from_numpy((np.random.uniform(size=len(index1s))*0.4 - 0.2)).to(lambdas.device)
        lambdas = lambdas + noises # add small random noises ~ U(-0.2, 0.2)
        lambdas = torch.clamp(lambdas, min=0.2, max=0.8) # clamp

        return lambdas

    @staticmethod
    def reweight_lambdas(index1s, index2s, T, IP):
        C1, C2 = T[index1s], T[index2s]
        X1P1 = torch.clamp(IP[index1s, C1], min=0., max=1.)
        X2P2 =  torch.clamp(IP[index2s, C2], min=0., max=1.)
        lambdas = (X1P1 + (1-X2P2)) / 2 # lambda_best
        lambdas = torch.clamp(lambdas, min=0., max=1.) # clamp
        return lambdas

    def intracls_mixup(self, X, T, IP):

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(self.proxies), smoothing_const=0
        )  # (N, C)
        D2T = IP * T  # (N, C) sparse matrix
        non_empty_mask = D2T.abs().sum(dim=0).bool()
        T_sub = torch.where(non_empty_mask == True)[0]  # record the class labels
        T_sub = torch.repeat_interleave(T_sub, self.pairs_per_cls)

        D2T_normalize = D2T[:, non_empty_mask]  # (N, C_sub) filter out non-zero columns which are the classes not sampled in this batch
        D2T_normalize = masked_softmax(D2T_normalize, dim=0)  # normalize for each class to find highest confidence samples to mix (N, C_sub)

        # weighted sampling weighted by inverted confidence
        if self.sampling_method in [1,3]:
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(self.pairs_per_cls):
                    prob_vec = D2T_normalize[:, j].detach().cpu().numpy() # take j column
                    index_pool = np.where(prob_vec != 0)[0] # find the indices withint the batch that takes class label j
                    prob_vec = (1.- prob_vec[index_pool]) # inverse its probability

                    if np.sum(prob_vec == 0) == len(prob_vec): # by right, this should not happen
                        logging.info('All probabilities are zeros')
                        pair_ind = np.random.choice(index_pool, 2,
                                                    replace=False).tolist()
                    else:
                        pair_ind = np.random.choice(index_pool, 2,
                                                    p=(prob_vec + 1e-8) / (prob_vec.sum() + 1e-8), # numerical stability
                                                    replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long()  # (2, C_sub*pairs_percls)

        # uniform random sampling
        elif self.sampling_method == 2:
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(self.pairs_per_cls):
                    pair_ind = np.random.choice(D2T_normalize.size()[0], 2,
                                                replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long()  # (2, C_sub*pairs_percls)

        selectedX_samecls = torch.stack([X[index, :] for index in mixup_ind_samecls], dim=-1)  # of shape (C_sub*pairs_percls, sz_embed, 2)

        # sample lambda coefficients
        lambda_samecls = self.random_lambdas() * torch.ones((len(selectedX_samecls), self.sz_embed))
        lambda_samecls = torch.stack((lambda_samecls, 1.-lambda_samecls), dim=-1).to(selectedX_samecls.device)  # (C_sub*pairs_percls, sz_embed, 2)

        # perform MixUp
        virtual_samples = torch.sum(selectedX_samecls * lambda_samecls, dim=-1)  # (C_sub*pairs_percls, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)
        assert T_sub.size()[0] == virtual_samples.size()[0]
        return T_sub, virtual_samples

    def intercls_mixup(self, X, T, IP):
        # we only sythesize samples and interpolating class labels
        # get some inter-class pairs to mixup
        index1s = torch.arange(X.size()[0])
        index2s = torch.roll(index1s, -self.shifts)  # shifts should be equal to n_samples_per_cls in your balanced sampler
        index1s, index2s = index1s.long(), index2s.long()
        cls_index1s, cls_index2s = T[index1s], T[index2s]

        if self.sampling_method == 1:
            # uncertainty-based sampling
            lambdas_diffcls = self.uncertainty_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)
        elif self.sampling_method == 2:
            # pure random sampling
            lambdas_diffcls = self.random_lambdas() * torch.ones((len(index1s), self.sz_embed))
        elif self.sampling_method == 3:
            lambdas_diffcls = self.reweight_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)

        # virtual samples
        selectedX_diffcls = torch.stack([X[index1s, :], X[index2s, :]], dim=-1)  # of shape (n_samples, sz_embed, 2)
        lambdas_diffcls = torch.stack((lambdas_diffcls, 1.-lambdas_diffcls), dim=-1).to(selectedX_diffcls.device)  # (n_samples, sz_embed, 2)
        virtual_samples = torch.sum(selectedX_diffcls * lambdas_diffcls, dim=-1)  # (n_samples, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)

        # interpolating class labels
        C1 = binarize_and_smooth_labels(T=cls_index1s, nb_classes=self.nb_classes, smoothing_const=0)  # (n_samples, C)
        C2 = binarize_and_smooth_labels(T=cls_index2s, nb_classes=self.nb_classes, smoothing_const=0)  # (n_samples, C)
        C = torch.stack((C1, C2), dim=-1)  # (n_samples, C, 2)
        lambdas_diffcls = lambdas_diffcls[:, :self.nb_classes, :]  # same lambdas used as when creating virtual proxies
        virtual_classes = torch.sum(C * lambdas_diffcls, dim=-1)  # (n_samples, C)

        assert virtual_classes.size()[0] == virtual_samples.size()[0]
        return virtual_classes, virtual_samples

    def intercls_mixup_proxy(self, X, T, P, IP):
        # We synthesize proxies as well as samples, where the new proxies are treated as new classes
        # get some inter-class pairs to mixup
        index1s = torch.arange(X.size()[0])
        index2s = torch.roll(index1s, -self.shifts)  # shifts should be equal to n_samples_per_cls in your balanced sampler?
        index1s, index2s = index1s.long(), index2s.long()
        cls_index1s, cls_index2s = T[index1s], T[index2s]
        _, counts = torch.unique_consecutive(cls_index1s, return_counts=True)

        # get mixup coefficients
        if self.sampling_method == 1:
            # uncertainty-based
            lambdas_diffcls = self.uncertainty_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)
        elif self.sampling_method == 2:
            # random
            lambdas_diffcls = self.random_lambdas() * torch.ones((len(index1s), self.sz_embed))
        elif self.sampling_method == 3:
            # reweight
            lambdas_diffcls = self.reweight_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)

        lambdas_diffcls = torch.stack((lambdas_diffcls, 1.-lambdas_diffcls), dim=-1).to(X.device)  # (n_samples, sz_embed, 2)

        # virtual samples
        selectedX_diffcls = torch.stack([X[index1s, :], X[index2s, :]], dim=-1)  # of shape (n_samples, sz_embed, 2)
        virtual_samples = torch.sum(selectedX_diffcls * lambdas_diffcls, dim=-1)  # (n_samples, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)

        P_pairs = torch.stack((P[cls_index1s, :], P[cls_index2s, :]), dim=-1)  # (n_samples, sz_embed, 2)
        if self.sampling_method in [1, 3]:
            virtual_proxies = torch.sum(P_pairs * 0.5, dim=-1)  # (n_samples, sz_embed)  virtual proxies with mix ratio 0.5
        elif self.sampling_method == 2:
            virtual_proxies = torch.sum(P_pairs * lambdas_diffcls, dim=-1)  # (n_samples, sz_embed)

        virtual_proxies = F.normalize(virtual_proxies, p=2, dim=-1)  # (n_samples, sz_embed)
        virtual_proxies = virtual_proxies[torch.cumsum(counts, dim=0) - counts[0], :]  # remove repeated entries

        #  virtual class labels (give new labels)
        virtual_classes = torch.arange(self.nb_classes,
                                       self.nb_classes + virtual_proxies.size()[0]).to(virtual_proxies.device)
        virtual_classes = virtual_classes.unsqueeze(-1)
        virtual_classes = torch.repeat_interleave(virtual_classes, counts, dim=0)  # repeat
        virtual_classes = virtual_classes.squeeze(-1)

        assert len(virtual_classes) == len(virtual_samples)
        return virtual_classes, virtual_samples, virtual_proxies

    def forward(self, X, indices, T):

        P = self.proxies
        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D, IP = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )
        D = D[:X.size()[0], X.size()[0]:] # (N, C)
        IP = IP[:X.size()[0], X.size()[0]:] # (N, C)

        if self.mixup_method == 'none':
            # no MixUp applied
            T = binarize_and_smooth_labels(
                T=T, nb_classes=len(P), smoothing_const=0
            ) # (N, C)

            loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
            loss = loss.mean()

        elif self.mixup_method == 'intra':
            # intra-class
            T_sub, virtual_X = self.intracls_mixup(X, T, IP, pairs_per_cls=4)
            Xall = torch.cat((X, self.scale * virtual_X), dim=0)
            Tall = torch.cat([T, T_sub])
            Dall = pairwise_distance(
                torch.cat(
                    [Xall, P]
                ),
                squared=True
            )[0][: Xall.size()[0], Xall.size()[0]:]

            Tall = binarize_and_smooth_labels(
                T=Tall, nb_classes=len(P), smoothing_const=0
            )
            loss = torch.sum(- Tall * F.log_softmax(-Dall, -1), -1)
            loss = loss.mean()

        elif self.mixup_method == 'inter_noproxy':
            # interclass without synthetic proxy
            virtual_classes, virtual_samples = self.intercls_mixup(X, T, IP)
            Xall = torch.cat((X, self.scale * virtual_samples), dim=0) # (N+N_inter, sz_embed)
            T = binarize_and_smooth_labels(
               T=T, nb_classes=len(P), smoothing_const=0
            ) # (N, C)
            Tall = torch.cat([T, virtual_classes], 0) # (N+N_inter, C)

            Dall = pairwise_distance(
                    torch.cat(
                        [Xall, P]
                    ),
                    squared=True
            )[0][: Xall.size()[0], Xall.size()[0]:] # (N+N_inter, C)

            loss = torch.sum(- Tall * F.log_softmax(-Dall, -1), -1)
            loss = loss.mean()

        elif self.mixup_method == 'inter_proxy':
            # interclass with synthetic proxy
            virtual_classes, virtual_samples, virtual_proxies = self.intercls_mixup_proxy(X, T, P, IP)
            Xall = torch.cat((X, self.scale * virtual_samples), dim=0) # (N+N_inter, sz_embed)
            Tall = torch.cat([T, virtual_classes], 0) # (N+N_inter,)
            Pall = torch.cat([P, self.scale * virtual_proxies]) # (C+C', sz_embed)

            Tall = binarize_and_smooth_labels(
               T=Tall, nb_classes=len(Pall), smoothing_const=0
            ) # (N+N_inter, C+C')
            Dall = pairwise_distance(
                    torch.cat(
                        [Xall, Pall]
                    ),
                    squared=True
            )[0][: Xall.size()[0], Xall.size()[0]:] # (N+N_inter, C+C')

            loss = torch.sum(- Tall * F.log_softmax(-Dall, -1), -1)
            loss = loss.mean()

        elif self.mixup_method == 'both':
            # Both intra-class and inter-class
            # between same class
            virtual_classes_intra, virtual_samples_intra = self.intracls_mixup(X, T, IP)
            # between dfferent class with proxy
            virtual_classes_inter, virtual_samples_inter, virtual_proxies_inter = self.intercls_mixup_proxy(X, T, P, IP)

            Xall = torch.cat((X, self.scale * virtual_samples_inter, self.scale * virtual_samples_intra), dim=0) # (N+N_inter+N_intra, sz_embed)
            Tall = torch.cat([T, virtual_classes_inter, virtual_classes_intra], 0) # (N+N_inter+N_intra, )
            Pall = torch.cat([P, self.scale * virtual_proxies_inter])  # (C+C_inter, sz_embed)

            Tall = binarize_and_smooth_labels(
               T=Tall, nb_classes=len(Pall), smoothing_const=0
            ) # (N+N_inter+N_intra, C+C_inter)

            Dall = pairwise_distance(
                    torch.cat(
                        [Xall, Pall]
                    ),
                    squared=True
            )[0][: Xall.size()[0], Xall.size()[0]:] # (N+N_inter+N_intra, C+C_inter)

            loss = torch.sum(- Tall * F.log_softmax(-Dall, -1), -1)
            loss = loss.mean()

        else:
            raise NotImplementedError

        return loss

class ProxyNCA_prob_pregularizer(torch.nn.Module):
    '''
        Original loss in ProxyNCA++
    '''
    def __init__(self, nb_classes, sz_embed, scale, regular_strength, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale
        self.regular_strength = regular_strength

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

        # regularization loss on inter proxy, use inner product because euclidean distance might produce high gradient (givent proxy_lr is high, better to choos elow gradient for it)
        P_norm = F.normalize(self.proxies, p=2, dim=-1)
        inter_proxy_IP = torch.mm(P_norm, torch.t(P_norm))
        inter_proxy_IP_mean = torch.triu(inter_proxy_IP, diagonal=1).mean()
        loss_all = loss + self.regular_strength * inter_proxy_IP_mean

        return loss_all


class ProxyNCA_prob_match(torch.nn.Module):

    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale

    def cross_attention(self, X, T, P):
        '''
            Cross attention calculation, give more weightage to neighbors closer to (corresponding) decision boundary or closer to the anchor itself
        '''
        P_batch = P[T.long(), :] # ground-truth proxies (N, sz_embed)
        decision_boundaries = (P_batch.unsqueeze(0) + P_batch.unsqueeze(1)) / 2 # (N, N, sz_embed) decision boundaries (middle point between 2 proxies)
        affinity2boundary = F.relu(torch.einsum("bkj,kj->bk", decision_boundaries, X)) # boundary-to-data affinity, (N, N)
        affinity2x = F.relu(torch.einsum("bj,kj->bk", X, X)) # data-to-data affinity (N, N)

        return affinity2boundary + affinity2x

    def forward(self, X, indices, T):
        P = self.proxies
        P = F.normalize(P, p=2, dim=-1)
        X = F.normalize(X, p=2, dim=-1)

        # self-attention update like simplified message passing network
        attention = self.cross_attention(X, T, P) # (N, N)
        attention = attention / (attention.sum(-1) + 1e-5) # normalize over batch (N, N), each row is the importance of all other intra-batch samples relative to this sample
        attention_weights = torch.einsum("bk,kj->bj", attention, X) # weighted average of neighbors (N, sz_embed)
        X = X + attention_weights # weighted average + residual connection

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

class ProxyNCA_pfix(torch.nn.Module):
    '''
        Original loss in ProxyNCA++
    '''
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed)) # not training
        self.proxies.requires_grad = False
        self._proxy_init(nb_classes, sz_embed)
        self.scale = scale

    def _proxy_init(self, nb_classes, sz_embed):
        proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
        _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
        for _ in range(100):
            mean, var = utils.inter_proxy_dist(proxies, cosine=True)
            _loss = mean + var
            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()

        proxies = F.normalize(proxies, p=2, dim=-1)
        self.proxies.data = proxies.detach()

    @torch.no_grad()
    def assign_cls4proxy(self, cls_mean):
        cls2proxy = torch.einsum('bi,mi->bm', cls_mean, self.proxies) # class mean to proxy affinity
        row_ind, col_ind = linear_sum_assignment((1-cls2proxy.detach().cpu()).numpy()) # row_ind: which class, col_ind: which proxy
        cls_indx = row_ind.argsort()
        sorted_class = row_ind[cls_indx]
        sorted_proxies = col_ind[cls_indx]
        self.proxies.data = self.proxies[sorted_proxies]
        logging.info('Number of updated proxies: {}'.format(np.sum(sorted_proxies != np.asarray(range(len(self.proxies))))))

    def forward(self, X, indices, T):
        P = self.proxies
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
