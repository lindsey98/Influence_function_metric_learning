"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch.nn as nn
from deprecated.AlignMix.networks import Generator
from deprecated.AlignMix.similarity import pairwise_distance
import sklearn.preprocessing
from deprecated.AlignMix.utils import *
from tqdm import tqdm

def recon_criterion(predict, target):
    '''
        Use L2 reconstruction loss
    '''
    return torch.mean(torch.square(predict - target))

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



class AlignMixModel(nn.Module):

    def __init__(self, hp):
        super(AlignMixModel, self).__init__()
        self.gen = Generator(hp['gen']) # generater
        self.gen_test = copy.deepcopy(self.gen)
        self.proxies = torch.nn.Parameter(torch.randn(hp['nb_classes'], hp['sz_embed']) / 8)
        self.scale = hp['scale']

    def calc_metric_learning_loss(self, X, T):
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

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-1
        max_iter = 100
        for i in range(max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    def normalize_all(self, x, y, x_mean, y_mean):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        x_mean = F.normalize(x_mean, dim=1)
        y_mean = F.normalize(y_mean, dim=1)
        return x, y, x_mean, y_mean

    def pair_wise_wdist(self, x, y):
        B, C, H, W = x.size()
        x = x.view(B, C, -1, 1)
        y = y.view(B, C, 1, -1)
        x_mean = x.mean([2, 3])
        y_mean = y.mean([2, 3])

        x, y, x_mean, y_mean = self.normalize_all(x, y, x_mean, y_mean)
        dist1 = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=1) + 1e-6).view(B, H * W, H * W)

        x = x.view(B, C, -1) # flatten
        y = y.view(B, C, -1)

        sim = torch.einsum('bcs, bcm->bsm', x, y).contiguous()
        # if self.use_uniform:
        u = torch.zeros(B, H * W, dtype=sim.dtype, device=sim.device).fill_(1. / (H * W))
        v = torch.zeros(B, H * W, dtype=sim.dtype, device=sim.device).fill_(1. / (H * W))
        wdist = 1.0 - sim.view(B, H * W, H * W)
        eps = 0.05

        with torch.no_grad():
            K = torch.exp(-wdist / eps)
            T = self.Sinkhorn(K, u, v)

        if torch.isnan(T).any():
            return None

        dist = torch.sum(T * dist1, dim=(1, 2))

        return dist, T

    def sample_lambdas(self, alpha=1.):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        return lam

    def forward(self, xa, la, xb, lb):
        xa = xa.cuda() # content image
        la = la.cuda() # content image's class label
        xb = xb.cuda()  # content image
        lb = lb.cuda()  # content image's class label

        c_xa, feat_xa = self.gen.encode(xa) # call content encoder
        xa_hat = self.gen.decode(c_xa)
        c_xb, feat_xb = self.gen.encode(xb)  # call content encoder
        xb_hat = self.gen.decode(c_xb)

        ### Structural matching mixup
        struct_dist, T = self.pair_wise_wdist(xa, xb) # T is of shape B, H*W, H*W
        xa_flatten = F.normalize(xa, dim=1).reshape(xa.size()[0], xa.size()[1], -1) # B, C, H*W
        xb_flatten = F.normalize(xb, dim=1).reshape(xb.size()[0], xb.size()[1], -1) # B, C, H*W

        xb_tomix = torch.einsum('bmk,bji->bmi', xa_flatten, T)
        xa_tomix = torch.einsum('bmk,bji->bmi', xb_flatten, T.permute(0,2,1))
        lam = self.sample_lambdas()

        mixed_xa = lam * xa + (1-lam) * xa_tomix
        mixed_xb = lam * xb + (1-lam) * xb_tomix
        mixed_xa = mixed_xa.reshape(xa.size()[0], xa.size()[1], xa.size()[2], xa.size()[3])
        mixed_xb = mixed_xb.reshape(xb.size()[0], xb.size()[1], xb.size()[2], xb.size()[3])

        ######## Reconstruction Loss ##################
        l_x_rec_a = recon_criterion(xa_hat, xa)
        l_x_rec_b = recon_criterion(xb_hat, xb)

        ######## Classification loss ##################
        feat_mixeda = F.adaptive_avg_pool2d(mixed_xa, (1,1))
        feat_mixedb = F.adaptive_avg_pool2d(mixed_xb, (1,1))

        l_c_rec_a = self.calc_metric_learning_loss(feat_xa, la) # xa
        l_c_rec_b = self.calc_metric_learning_loss(feat_xb, lb) # xa
        l_c_rec_mixeda = lam * self.calc_metric_learning_loss(feat_mixeda, la) + \
                         (1-lam) * self.calc_metric_learning_loss(feat_mixeda, lb)
        l_c_rec_mixedb = lam * self.calc_metric_learning_loss(feat_mixedb, lb) + \
                         (1-lam) * self.calc_metric_learning_loss(feat_mixedb, la)

        l_total = l_x_rec_a + l_x_rec_b + l_c_rec_a + l_c_rec_b + l_c_rec_mixeda + l_c_rec_mixedb
        l_total.backward()
        return l_total, l_x_rec_a, l_x_rec_b, l_c_rec_a, l_c_rec_b, l_c_rec_mixeda, l_c_rec_mixedb

    @torch.no_grad()
    def predict_batchwise(self, dataloader):
        self.gen.eval()
        X = torch.tensor([])
        T = torch.tensor([])
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc="Batch-wise prediction"):
            x, y, _ = batch
            # move images to device of model (approximate device)
            x, y = x.cuda(), y.cuda()
            # predict model output for image
            _, feat = self.gen.encode(x)
            feat = feat.view(feat.size()[0], -1)
            X = torch.cat((X, feat.detach().cpu()), dim=0)
            T = torch.cat((T, y.detach().cpu()), dim=0)
        self.gen.train()
        print(X.shape)
        print(T.shape)

        return X, T

    @torch.no_grad()
    def evaluate(self, dataloader,
                 eval_nmi=True, recall_list=[1, 2, 4, 8]):

        eval_time = time.time()
        nb_classes = len(dataloader.dataset.classes)
        # calculate embeddings with model and get targets
        X, T = self.predict_batchwise(dataloader)
        print('done collecting prediction')

        if eval_nmi:
            # calculate NMI with kmeans clustering
            nmi = calc_normalized_mutual_information(
                T, cluster_by_kmeans(X, nb_classes)
            )
        else:
            nmi = 1

        print("NMI: {:.3f}".format(nmi * 100))

        # get predictions by assigning nearest 8 neighbors with euclidian
        max_dist = max(recall_list)
        Y = assign_by_euclidian_at_k(X, T, max_dist)
        Y = torch.from_numpy(Y)

        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in recall_list:
            r_at_k = calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            print("R@{} : {:.3f}".format(k, 100 * r_at_k))

        chmean = (2 * nmi * recall[0]) / (nmi + recall[0])
        print("hmean: %s", str(chmean))

        eval_time = time.time() - eval_time
        logging.info('Eval time: %.2f' % eval_time)
        return nmi, recall

    @torch.no_grad()
    def test(self, xa, la):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()

        ## The following are produced by the current trained generator
        c_xa_current, feat_xa_current = self.gen.encode(xa)
        xt_current = self.gen.decode(c_xa_current)

        ## The following are produced by the initial generator before training
        c_xa, feat_xa = self.gen_test.encode(xa)
        xt = self.gen_test.decode(c_xa)

        self.train()
        return xa, xt_current, xt

