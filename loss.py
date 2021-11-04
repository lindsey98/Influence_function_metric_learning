import torch

from similarity import pairwise_distance
import torch.nn.functional as F
import math
from typing import Union
import sklearn.preprocessing
from deprecated.hard_sample_detection.gmm import *

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

    def __init__(self, nb_classes, sz_embed, scale, mixup_method, sampling_method, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale
        self.sz_embed = sz_embed
        self.nb_classes = nb_classes
        self.mixup_method = mixup_method
        self.sampling_method = sampling_method
        assert self.mixup_method in ['none', 'intra', 'inter_noproxy', 'inter_proxy', 'both']
        assert self.sampling_method in [1, 2] # 1 is weighted sampling, 2 is random sampling

    @staticmethod
    def random_lambdas(n_samples, alpha=1.0):
        lambdas = []
        for _ in range(n_samples):
            lambdas.append(np.random.beta(alpha, alpha)) # alpha=beta=1 is uniform distribution
        return torch.tensor(lambdas)

    @staticmethod
    def uncertainty_lambdas(index1s, index2s, T, IP):
        C1, C2 = T[index1s], T[index2s]
        X1P1, X1P2 = torch.clamp(IP[index1s, C1], min=-1., max=1.), \
                     torch.clamp(IP[index1s, C2], min=-1., max=1.) # (n_samples, )
        X2P1, X2P2 = torch.clamp(IP[index2s, C1], min=-1., max=1.), \
                     torch.clamp(IP[index2s, C2], min=-1., max=1.)
        lambdas = (X2P2-X2P1) / ((X2P2-X2P1) + (X1P1-X1P2))
        lambdas = lambdas + torch.from_numpy((np.random.uniform(size=len(index1s))*0.4-0.2)).to(lambdas.device) # add small random noises e~unif(-0.2, 0.2)
        lambdas = torch.clamp(lambdas, min=0.2, max=0.8) # clamp

        return lambdas

    def intracls_mixup(self, X, T, IP, pairs_per_cls):

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(self.proxies), smoothing_const=0
        )  # (N, C)
        D2T = IP * T  # (N, C) sparse matrix
        non_empty_mask = D2T.abs().sum(dim=0).bool()
        T_sub = torch.where(non_empty_mask == True)[0]  # record the class labels
        T_sub = torch.repeat_interleave(T_sub, pairs_per_cls)

        D2T_normalize = D2T[:, non_empty_mask]  # (N, C_sub) filter out non-zero columns which are the classes not sampled in this batch
        D2T_normalize = masked_softmax(D2T_normalize, dim=0)  # normalize for each class to find highest confidence samples to mix (N, C_sub)

        # weighted sampling weighted by inverted confidence
        if self.sampling_method == 1:
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(pairs_per_cls):
                    prob_vec = D2T_normalize[:, j].detach().cpu().numpy()
                    prob_vec = ((1.-prob_vec) * (prob_vec != 0)) / np.sum((1.-prob_vec) * (prob_vec != 0))
                    pair_ind = np.random.choice(D2T_normalize.size()[0], 2,
                                                p=prob_vec, # inverse probability
                                                replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long()  # (2, C_sub*pairs_percls)

        # uniform random sampling
        elif self.sampling_method == 2:
            mixup_ind_samecls = torch.tensor([])
            for j in range(D2T_normalize.size()[1]):
                for _ in range(pairs_per_cls):
                    pair_ind = np.random.choice(D2T_normalize.size()[0], 2,
                                                replace=False).tolist()
                    mixup_ind_samecls = torch.cat((mixup_ind_samecls, torch.tensor([pair_ind]).T), dim=-1)
            mixup_ind_samecls = mixup_ind_samecls.long()  # (2, C_sub*pairs_percls)

        selectedX_samecls = torch.stack([X[index, :] for index in mixup_ind_samecls],
                                        dim=-1)  # of shape (C_sub*pairs_percls, sz_embed, 2)
        # sample lambda coefficients (different lambda for different samples)
        lambda_samecls = self.random_lambdas(selectedX_samecls.size()[0])
        lambda_samecls = lambda_samecls.unsqueeze(-1).repeat(1, self.sz_embed)
        lambda_samecls = torch.stack((lambda_samecls, 1.-lambda_samecls), dim=-1).to(
            selectedX_samecls.device)  # (C_sub*pairs_percls, sz_embed, 2)

        # perform MixUp
        virtual_samples = torch.sum(selectedX_samecls * lambda_samecls, dim=-1)  # (C_sub*pairs_percls, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)
        assert T_sub.size()[0] == virtual_samples.size()[0]
        return T_sub, virtual_samples

    def intercls_mixup(self, X, T, IP, shifts=4):
        # we only sythesize samples and interpolating class labels
        # get some inter-class pairs to mixup
        index1s = torch.arange(X.size()[0])
        index2s = torch.roll(index1s, -shifts)  # shifts should be equal to n_samples_per_cls in your balanced sampler
        index1s, index2s = index1s.long(), index2s.long()
        cls_index1s, cls_index2s = T[index1s], T[index2s]

        if self.sampling_method == 1:
            # uncertainty-based sampling
            lambdas_diffcls = self.uncertainty_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)
        elif self.sampling_method == 2:
            # pure random sampling
            lambdas_diffcls = self.random_lambdas(len(index1s)).unsqueeze(-1).repeat(1, self.sz_embed)

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

    def intercls_mixup_proxy(self, X, T, P, IP, shifts=4):
        # We synthesize proxies as well as samples, where the new proxies are treated as new classes
        # get some inter-class pairs to mixup
        index1s = torch.arange(X.size()[0])
        index2s = torch.roll(index1s, -shifts)  # shifts should be equal to n_samples_per_cls in your balanced sampler
        index1s, index2s = index1s.long(), index2s.long()
        cls_index1s, cls_index2s = T[index1s], T[index2s]

        # virtual proxies with mix ratio 0.5
        P_pairs = torch.stack((P[cls_index1s, :], P[cls_index2s, :]), dim=-1)  # (n_samples, sz_embed, 2)
        virtual_proxies = torch.sum(P_pairs * 0.5, dim=-1)  # (n_samples, sz_embed)
        virtual_proxies = F.normalize(virtual_proxies, p=2, dim=-1)  # (n_samples, sz_embed)
        virtual_proxies = virtual_proxies[torch.arange(0, len(virtual_proxies), shifts), :]  # remove repeated entries

        #  virtual class labels (give new labels)
        virtual_classes = torch.arange(self.nb_classes, self.nb_classes + virtual_proxies.size()[0]).to(virtual_proxies.device)
        virtual_classes = torch.repeat_interleave(virtual_classes, shifts)  # repeat

        if self.sampling_method == 1:
            # uncertainty-based
            lambdas_diffcls = self.uncertainty_lambdas(index1s, index2s, T, IP).unsqueeze(-1).repeat(1, self.sz_embed)
        elif self.sampling_method == 2:
            # random
            lambdas_diffcls = self.random_lambdas(len(index1s)).unsqueeze(-1).repeat(1, self.sz_embed)
        lambdas_diffcls = torch.stack((lambdas_diffcls, 1.-lambdas_diffcls), dim=-1).to(X.device)  # (n_samples, sz_embed, 2)

        # virtual samples
        selectedX_diffcls = torch.stack([X[index1s, :], X[index2s, :]], dim=-1)  # of shape (n_samples, sz_embed, 2)
        virtual_samples = torch.sum(selectedX_diffcls * lambdas_diffcls, dim=-1)  # (n_samples, sz_embed)
        virtual_samples = F.normalize(virtual_samples, p=2, dim=-1)

        assert virtual_classes.size()[0] == virtual_samples.size()[0]
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
            virtual_classes_intra, virtual_samples_intra = self.intracls_mixup(X, T, IP, pairs_per_cls=4)
            # between dfferent class with proxy
            virtual_classes_inter, virtual_samples_inter, virtual_proxies_inter = self.intercls_mixup_proxy(X, T, P, IP)

            Xall = torch.cat((X,
                                self.scale * virtual_samples_inter,
                                self.scale * virtual_samples_intra), dim=0) # (N+N_inter+N_intra, sz_embed)
            Tall = torch.cat([T,
                                virtual_classes_inter,
                                virtual_classes_intra], 0) # (N+N_inter+N_intra, )
            Pall = torch.cat([P,
                                self.scale * virtual_proxies_inter])  # (C+C_inter, sz_embed)

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




