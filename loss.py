import numpy as np
import torch
from similarity import pairwise_distance
import torch.nn.functional as F
import sklearn.preprocessing
import logging
from scipy.optimize import linear_sum_assignment
import utils
from tqdm import tqdm

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

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed,
                 mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        torch.nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, indices, T):
        P = self.proxies
        P = F.normalize(P, p=2, dim=-1)
        X = F.normalize(X, p=2, dim=-1)

        cos = F.linear(X, P)  # Calcluate cosine similarity
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

        return loss

class ProxyNCA_prob_orig(torch.nn.Module):
    '''
        Original loss in ProxyNCA++
    '''
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    @torch.no_grad()
    def debug(self, X, indices, T):
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
        return loss, None

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

class ProxyNCA_pfix(torch.nn.Module):
    '''
        ProxyNCA++ with fixed proxies
    '''
    def __init__(self, nb_classes, sz_embed, scale, initialize_method='optim', super_classes=None, cls_mean=None, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed)) # not training
        self.proxies.requires_grad = False
        self.scale = scale
        self.initialize_method = initialize_method
        self.super_classes = super_classes
        self.cls_mean = cls_mean
        self._proxy_init(nb_classes, sz_embed)

    def _proxy_init(self, nb_classes, sz_embed):
        if self.initialize_method == 'optim':
            proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
            _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
            for _ in tqdm(range(100), desc="Initializing the proxies"):
                mean, var = utils.inter_proxy_dist(proxies)
                _loss = mean + var
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

            proxies = F.normalize(proxies, p=2, dim=-1)
            self.proxies.data = proxies.detach()
        elif self.initialize_method == 'random':
            pass # do nothing
        elif self.initialize_method == 'duplicate':
            dup_cls = np.random.choice(nb_classes, int(nb_classes/5), replace=False)
            for cls in dup_cls:
                if cls == 0:
                    self.proxies.data[cls] = self.proxies.data[cls+1]
                else:
                    self.proxies.data[cls] = self.proxies.data[cls-1]

        elif self.initialize_method == 'super_optim':
            proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
            proxies.data = self.cls_mean.to(proxies.device)
            _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
            for _ in tqdm(range(10), desc="Initializing the proxies"):
                mean, var, _ = utils.inter_proxy_dist_super(proxies, self.super_classes)
                _loss = mean + var
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

            proxies = F.normalize(proxies, p=2, dim=-1)
            self.proxies.data = proxies.detach()


    @torch.no_grad()
    def debug(self, X, indices, T):
        P = self.scale * F.normalize(self.proxies, p=2, dim=-1)
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

        prob = F.softmax(-D, -1) # (N, C)
        values, _ = torch.topk(prob, k=2, dim=-1)# top1 - top2
        margin = values[:, 0] - values[:, 1]
        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        return loss, margin

    @torch.no_grad()
    def assign_cls4proxy(self, cls_mean):
        cls2proxy = torch.einsum('bi,mi->bm', cls_mean, self.proxies) # class mean to proxy affinity
        row_ind, col_ind = linear_sum_assignment((1-cls2proxy.detach().cpu()).numpy()) # row_ind: which class, col_ind: which proxy
        cls_indx = row_ind.argsort() # class from 1, 2 ... C
        sorted_class = row_ind[cls_indx]
        sorted_proxies = col_ind[cls_indx] # proxy indices correponding to class from 1, 2 ... C
        self.proxies.data = self.proxies[sorted_proxies] # sort proxies according to proxy indices
        logging.info('Number of updated proxies: {}'.format(np.sum(sorted_proxies != np.asarray(range(len(self.proxies))))))

    def forward_score(self, X, T):
        P = self.proxies; P.requires_grad = False
        T.requires_grad = False
        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[0][:X.size()[0], X.size()[0]:]

        scores = F.log_softmax(-D, -1) # (N, C)
        loss = scores[torch.arange(len(T)), T.long()]
        return loss

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

class ProxyNCA_pfix_reweight(torch.nn.Module):
    '''
        ProxyNCA++ with fixed proxies
    '''
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed)) # not training
        self.proxies.requires_grad = False
        self._proxy_init(nb_classes, sz_embed)
        self.scale = scale

    def _proxy_init(self, nb_classes, sz_embed): # TODO: random initialization or duplicate initialization
        proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
        _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
        for _ in tqdm(range(100), desc="Initializing the proxies"):
            mean, var = utils.inter_proxy_dist(proxies, cosine=True)
            _loss = mean + var
            _optimizer.zero_grad()
            _loss.backward()
            _optimizer.step()

        proxies = F.normalize(proxies, p=2, dim=-1)
        self.proxies.data = proxies.detach()

    @torch.no_grad()
    def debug(self, X, indices, T, weights):
        P = self.scale * F.normalize(self.proxies, p=2, dim=-1)
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

        prob = F.softmax(-D, -1) # (N, C)
        values, _ = torch.topk(prob, k=2, dim=-1)# top1 - top2
        margin = values[:, 0] - values[:, 1]
        loss = weights * torch.sum(- T * F.log_softmax(-D, -1), -1)
        return loss, margin

    @torch.no_grad()
    def assign_cls4proxy(self, cls_mean):
        cls2proxy = torch.einsum('bi,mi->bm', cls_mean, self.proxies) # class mean to proxy affinity
        row_ind, col_ind = linear_sum_assignment((1-cls2proxy.detach().cpu()).numpy()) # row_ind: which class, col_ind: which proxy
        cls_indx = row_ind.argsort() # class from 1, 2 ... C
        sorted_class = row_ind[cls_indx]
        sorted_proxies = col_ind[cls_indx] # proxy indices correponding to class from 1, 2 ... C
        self.proxies.data = self.proxies[sorted_proxies] # sort proxies according to proxy indices
        logging.info('Number of updated proxies: {}'.format(np.sum(sorted_proxies != np.asarray(range(len(self.proxies))))))


    def forward(self, X, indices, T, weights):
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
        # print(weights)
        loss = weights * torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss

class ProxyNCA_pfix_softlabel(torch.nn.Module):
    '''
        ProxyNCA++ with fixed proxies
    '''
    def __init__(self, nb_classes, sz_embed, scale, initialize_method='optim', super_classes=None, cls_mean=None, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed)) # not training
        self.proxies.requires_grad = False
        self.scale = scale
        self.initialize_method = initialize_method
        self.super_classes = super_classes
        self.cls_mean = cls_mean
        self._proxy_init(nb_classes, sz_embed)

    def _proxy_init(self, nb_classes, sz_embed):
        if self.initialize_method == 'optim':
            proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
            _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
            for _ in tqdm(range(100), desc="Initializing the proxies"):
                mean, var = utils.inter_proxy_dist(proxies)
                _loss = mean + var
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

            proxies = F.normalize(proxies, p=2, dim=-1)
            self.proxies.data = proxies.detach()
        elif self.initialize_method == 'random':
            pass # do nothing
        elif self.initialize_method == 'duplicate':
            dup_cls = np.random.choice(nb_classes, int(nb_classes/5), replace=False)
            for cls in dup_cls:
                if cls == 0:
                    self.proxies.data[cls] = self.proxies.data[cls+1]
                else:
                    self.proxies.data[cls] = self.proxies.data[cls-1]

        elif self.initialize_method == 'super_optim':
            proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
            proxies.data = self.cls_mean.to(proxies.device)
            _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
            for _ in tqdm(range(10), desc="Initializing the proxies"):
                mean, var, _ = utils.inter_proxy_dist_super(proxies, self.super_classes)
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
        cls_indx = row_ind.argsort() # class from 1, 2 ... C
        sorted_class = row_ind[cls_indx]
        sorted_proxies = col_ind[cls_indx] # proxy indices correponding to class from 1, 2 ... C
        self.proxies.data = self.proxies[sorted_proxies] # sort proxies according to proxy indices
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

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss

class ProxyAnchor_pfix(torch.nn.Module):
    '''
        Fixed anchor
    '''
    def __init__(self, nb_classes, sz_embed, initialize_method='optim', super_classes=None, cls_mean=None, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed)) # not training
        self.proxies.requires_grad = False
        self.initialize_method = initialize_method
        self._proxy_init(nb_classes, sz_embed, super_classes, cls_mean)
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def _proxy_init(self, nb_classes, sz_embed, super_classes, cls_mean):
        if self.initialize_method == 'optim':
            proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
            _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
            for _ in tqdm(range(100), desc="Initializing the proxies"):
                mean, var = utils.inter_proxy_dist(proxies, cosine=True)
                _loss = mean + var
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

            proxies = F.normalize(proxies, p=2, dim=-1)
            self.proxies.data = proxies.detach()
        elif self.initialize_method == 'random':
            pass # do nothing
        elif self.initialize_method == 'duplicate':
            dup_cls = np.random.choice(nb_classes, int(nb_classes/5), replace=False)
            for cls in dup_cls:
                if cls == 0:
                    self.proxies.data[cls] = self.proxies.data[cls+1]
                else:
                    self.proxies.data[cls] = self.proxies.data[cls-1]
        elif self.initialize_method == 'super_optim':
            proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
            proxies.data = cls_mean.to(self.proxies.device)
            _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
            for _ in tqdm(range(25), desc="Initializing the proxies"):
                mean, var = utils.inter_proxy_dist_super(proxies, super_classes)
                _loss = mean + var
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

            proxies = F.normalize(proxies, p=2, dim=-1)
            self.proxies.data = proxies.detach()
        else:
            raise NotImplementedError

    @torch.no_grad()
    def assign_cls4proxy(self, cls_mean):
        cls2proxy = torch.einsum('bi,mi->bm', cls_mean, self.proxies) # class mean to proxy affinity
        row_ind, col_ind = linear_sum_assignment((1-cls2proxy.detach().cpu()).numpy()) # row_ind: which class, col_ind: which proxy
        cls_indx = row_ind.argsort()
        sorted_class = row_ind[cls_indx]
        sorted_proxies = col_ind[cls_indx]
        self.proxies.data = self.proxies[sorted_proxies]
        logging.info('Number of updated proxies: {}'.format(np.sum(sorted_proxies != np.asarray(range(len(self.proxies))))))

    @torch.no_grad()
    def debug(self, X, indices, T):
        P = self.proxies
        P = F.normalize(P, p=2, dim=-1)
        X = F.normalize(X, p=2, dim=-1)

        cos = F.linear(X, P)  # Calcluate cosine similarity
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

        return loss, None

    def forward(self, X, indices, T):
        P = self.proxies
        P = F.normalize(P, p=2, dim=-1)
        X = F.normalize(X, p=2, dim=-1)

        cos = F.linear(X, P)  # Calcluate cosine similarity
        P_one_hot = binarize_and_smooth_labels(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg)) # (N, C)
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) # (C,)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss

class ProxyAnchor_pfix_reweight(torch.nn.Module):
    '''
        Fixed anchor
    '''
    def __init__(self, nb_classes, sz_embed, initialize_method='optim', mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed)) # not training
        self.proxies.requires_grad = False
        self.initialize_method = initialize_method
        self._proxy_init(nb_classes, sz_embed)
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def _proxy_init(self, nb_classes, sz_embed):
        if self.initialize_method == 'optim':
            proxies = torch.randn((nb_classes, sz_embed), requires_grad=True)
            _optimizer = torch.optim.Adam(params={proxies}, lr=0.1)
            for _ in tqdm(range(100), desc="Initializing the proxies"):
                mean, var = utils.inter_proxy_dist(proxies, cosine=True)
                _loss = mean + var
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

            proxies = F.normalize(proxies, p=2, dim=-1)
            self.proxies.data = proxies.detach()
        elif self.initialize_method == 'random':
            pass # do nothing
        elif self.initialize_method == 'duplicate':
            dup_cls = np.random.choice(nb_classes, int(nb_classes/5), replace=False)
            for cls in dup_cls:
                if cls == 0:
                    self.proxies.data[cls] = self.proxies.data[cls+1]
                else:
                    self.proxies.data[cls] = self.proxies.data[cls-1]
        else:
            raise NotImplementedError

    @torch.no_grad()
    def assign_cls4proxy(self, cls_mean):
        cls2proxy = torch.einsum('bi,mi->bm', cls_mean, self.proxies) # class mean to proxy affinity
        row_ind, col_ind = linear_sum_assignment((1-cls2proxy.detach().cpu()).numpy()) # row_ind: which class, col_ind: which proxy
        cls_indx = row_ind.argsort()
        sorted_class = row_ind[cls_indx]
        sorted_proxies = col_ind[cls_indx]
        self.proxies.data = self.proxies[sorted_proxies]
        logging.info('Number of updated proxies: {}'.format(np.sum(sorted_proxies != np.asarray(range(len(self.proxies))))))

        #TODO
        # self.proxies.data = cls_mean.to(self.proxies.device)
        # logging.info('Reassign proxies as class centers')

    @torch.no_grad()
    def debug(self, X, indices, T, weights):
        P = self.proxies
        P = F.normalize(P, p=2, dim=-1)
        X = F.normalize(X, p=2, dim=-1)

        cos = F.linear(X, P)  # Calcluate cosine similarity
        P_one_hot = binarize_and_smooth_labels(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = weights.unsqueeze(-1) * torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = weights.unsqueeze(-1) * torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss, None

    def forward(self, X, indices, T, weights):
        P = self.proxies
        P = F.normalize(P, p=2, dim=-1)
        X = F.normalize(X, p=2, dim=-1)

        cos = F.linear(X, P)  # Calcluate cosine similarity
        P_one_hot = binarize_and_smooth_labels(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = weights.unsqueeze(-1) * torch.exp(-self.alpha * (cos - self.mrg)) # (N, C)
        neg_exp = weights.unsqueeze(-1) * torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) # (C,)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


class SoftTriple(torch.nn.Module):
    def __init__(self, la, gamma, tau, margin, K,
                 nb_classes, sz_embed,
                 ):
        super(SoftTriple, self).__init__()
        self.la = la  # scaling factor in softmax loss
        self.gamma = 1. / gamma  # temperature scaling factor in q_k
        self.tau = tau  # tradeoff param in center regularization
        self.margin = margin  # delta
        self.nb_classes =  nb_classes
        self.sz_embed = sz_embed
        self.K = K
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes*K, sz_embed).cuda())
        self.weight = torch.zeros(nb_classes * K, nb_classes * K, dtype=torch.bool).cuda()
        for i in range(0, nb_classes):
            for j in range(0, K):
                self.weight[i * K + j, i * K + j + 1:(i + 1) * K] = 1
        torch.nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, X, indices, T):

        P = F.normalize(self.proxies, p=2, dim=-1)  # proxy embeddings
        X = F.normalize(X, p=2, dim=-1)
        simInd = X.matmul(P.t())  # (B, nb_classes*K)
        simStruc = simInd.reshape(-1, self.nb_classes, self.K) # (B, nb_classes, K)

        prob = F.softmax(simStruc * self.gamma, dim=2) # (B, nb_classes, K)
        simClass = torch.sum(prob * simStruc, dim=2) # (B, nb_classes)

        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), T] = self.margin # (B, nb_classes)
        loss = F.cross_entropy(self.la * (simClass-marginM), T) # scalar

        if self.tau > 0 and self.K > 1:
            simCenter = P.matmul(P.t()) # (nb_classes*K, nb_classes*K)
            reg = torch.sum(torch.sqrt(2.0 + 1e-5 - 2.*simCenter[self.weight])) / (self.nb_classes * self.K * (self.K-1.))
            return loss + self.tau * reg
        else:
            return loss