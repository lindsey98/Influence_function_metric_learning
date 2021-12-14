
from __future__ import print_function
from __future__ import division

import evaluation
import numpy as np
import torch
import logging
import json
import time
#import margin_net
import similarity
import torch.nn.functional as F
from tqdm import tqdm
import loss
import networks
from evaluation.map import *
from similarity import pairwise_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
# __repr__ may contain `\n`, json replaces it by `\\n` + indent
json_dumps = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def load_config(config_name = 'config.json'):
    '''
        Load config.json file
    '''
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)
    return config

def predict_batchwise(model, dataloader):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    # print(list(model.parameters())[0].device)
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc="Batch-wise prediction"):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).cpu()
                for j in J:
                    #if i == 1: print(j)
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]

@torch.no_grad()
def predict_batchwise_loss(model, dataloader, criterion):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    model.eval()
    base_loss = torch.tensor([])
    embeddings = torch.tensor([])
    labels = torch.tensor([])

    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for (x, y, indices) in tqdm(dataloader, desc="Batch-wise prediction"):
            x, y = x.to(list(model.parameters())[0].device), y.to(list(model.parameters())[0].device)
            m = model(x)
            loss, _ = criterion.debug(m, indices, y)
            base_loss = torch.cat((base_loss, loss.detach().cpu()), dim=0)
            embeddings = torch.cat((embeddings, m.detach().cpu()), dim=0)
            labels = torch.cat((labels, y.detach().cpu()), dim=0)

    return embeddings, labels, indices, base_loss

def predict_batchwise_inshop(model, dataloader):
    '''
        Predict on a batch on InShop dataset
        :param model:
        :param dataloader:
    '''
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # use tqdm when the dataset is large (SOProducts)
        is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in dataloader:#, desc='predict', disable=not is_verbose:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).data.cpu().numpy()
                    # take only subset of resulting embedding w.r.t dataset
                for j in J:
                    A[i].append(np.asarray(j))
        result = [np.stack(A[i]) for i in range(len(A))]
    model.train()
    model.train(model_is_training) # revert to previous training state
    return result

def evaluate(model, dataloader, eval_nmi=True, recall_list=[1, 2, 4, 8]):
    '''
        Evaluation on dataloader
        :param model: embedding model
        :param dataloader: dataloader
        :param eval_nmi: evaluate NMI (Mutual information between clustering on embedding and the gt class labels) or not
        :param recall_list: recall@K
    '''
    eval_time = time.time()
    nb_classes = len(dataloader.dataset.classes)

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dataloader)
    print('done collecting prediction')

    if eval_nmi:
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(
            T,
            evaluation.cluster_by_kmeans(
                X, nb_classes
            )
        )
    else:
        nmi = 1

    logging.info("NMI: {:.3f}".format(nmi))

    # Recall get predictions by assigning nearest 8 neighbors with euclidian
    max_dist = max(recall_list)
    Y = evaluation.assign_by_euclidian_at_k(X, T, max_dist)
    Y = torch.from_numpy(Y)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in recall_list:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    chmean = (2*nmi*recall[0]) / (nmi + recall[0])
    logging.info("hmean: %s", str(chmean))

    # MAP@R
    label_counts = get_label_match_counts(T, T) # get R
    # num_k = determine_k(
    #     num_reference_embeddings=len(T), embeddings_come_from_same_source=True
    # ) # equal to num_reference-1 (deduct itself)
    num_k = max([count[1] for count in label_counts])
    knn_indices = get_knn(
        X, X, num_k, True
    )
    knn_labels = T[knn_indices] # get KNN indicies
    map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                        gt_labels=T[:, None],
                                        embeddings_come_from_same_source=True,
                                        label_counts=label_counts,
                                        avg_of_avgs=False,
                                        label_comparison_fn=torch.eq)
    logging.info("MAP@R:{:.3f}".format(map_R * 100))

    eval_time = time.time() - eval_time
    logging.info('Eval time: %.2f' % eval_time)
    return nmi, recall, map_R


def evaluate_inshop(model, dl_query, dl_gallery,
                    K = [1, 10, 20, 30, 40, 50], with_nmi = True):
    '''
        Evaluate on Inshop dataset
    '''

    # calculate embeddings with model and get targets
    X_query, T_query, *_ = predict_batchwise(
        model, dl_query)
    X_gallery, T_gallery, *_ = predict_batchwise(
        model, dl_gallery)

    nb_classes = dl_query.dataset.nb_classes()

    assert nb_classes == len(set(T_query.detach().cpu().numpy()))

    # calculate full similarity matrix, choose only first `len(X_query)` rows
    # and only last columns corresponding to the column
    T_eval = torch.cat(
        [T_query, T_gallery])
    X_eval = torch.cat(
        [X_query, X_gallery])
    D = similarity.pairwise_distance(X_eval)[0][:X_query.size()[0], X_query.size()[0]:]

    # get top k labels with smallest (`largest = False`) distance
    Y = T_gallery[D.topk(k = max(K), dim = 1, largest = False)[1]]

    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T_query, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if with_nmi:
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(
            T_eval.numpy(),
            evaluation.cluster_by_kmeans(
                X_eval.numpy(), nb_classes
            )
        )
    else:
        nmi = 1

    logging.info("NMI: {:.3f}".format(nmi * 100))

    # MAP@R
    label_counts = get_label_match_counts(T_query, T_gallery) # get R
    # num_k = determine_k(
    #     num_reference_embeddings=len(T_gallery), embeddings_come_from_same_source=False
    # ) # equal to num_reference
    num_k = max([count[1] for count in label_counts])
    knn_indices = get_knn(
        X_gallery, X_query, num_k, True
    )
    knn_labels = T_gallery[knn_indices] # get KNN indicies
    map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                        gt_labels=T_query[:, None],
                                        embeddings_come_from_same_source=False,
                                        label_counts=label_counts,
                                        avg_of_avgs=False,
                                        label_comparison_fn=torch.eq)
    logging.info("MAP@R:{:.3f}".format(map_R * 100))


    return nmi, recall, map_R

def get_svd(model, dl, topk_singular=1, return_avg=False):
    X, T, *_ = predict_batchwise(model, dl) # get embedding
    singular_values = torch.tensor([])
    for cls in range(dl.dataset.nb_classes()):
        indices = T == cls
        X_cls = X[indices, :] # class-specific embedding
        u, s, v = torch.linalg.svd(X_cls) # compute singular value, lower value implies lower data variance
        s = s[:topk_singular] # only take top 5 singular values
        singular_values = torch.cat((singular_values, s.unsqueeze(0)), dim=0)

    if return_avg: # average over different classes or n
        singular_values = torch.mean(singular_values, dim=0)
    return singular_values


def bipartite_matching(embeddingX, embeddingY):

    D = pairwise_distance(
        torch.cat([embeddingX, embeddingY], dim=0)
    )[0][:len(embeddingX), len(embeddingX):] # (Nx, Ny)

    row_ind, col_ind = linear_sum_assignment(D.numpy())
    best_matchD = D[row_ind, col_ind]
    gapD = np.sort(best_matchD)[:min(10, len(best_matchD))].mean() # top10 edges distances
    return gapD


def calc_gap(model, dl, proxies, topk=5):
    X, T, *_ = predict_batchwise(model, dl)  # get embedding
    embeddings = []
    for cls in range(dl.dataset.nb_classes()):
        indices = T == cls
        X_cls = X[indices, :]  # class-specific embedding
        embeddings.append(X_cls)

    IP = pairwise_distance(proxies, squared=True)[1]
    _, knn_indices = torch.sort(IP, dim=-1, descending=True)
    knn_indices = knn_indices[:, 1:(topk+1)]

    gaps = np.zeros((knn_indices.size()[0], topk))
    for i in range(len(knn_indices)):
        for j in range(topk): # 5 NNs
            class_i = int(i)
            class_j = int(knn_indices[i][j].item())
            gaps[i][j] = bipartite_matching(embeddings[class_i].detach().cpu(),
                                            embeddings[class_j].detach().cpu())

    return gaps.mean()

def inter_proxy_dist(proxies, cosine=True, neighbor_only=False):
    D, IP = pairwise_distance(proxies, squared=True)
    if cosine:
        upper_triu = torch.triu(IP, diagonal=1)
    else:
        upper_triu = torch.triu(-D, diagonal=1)
    if not neighbor_only:
        reduced_mean = upper_triu.mean()
        reduced_std = torch.std(upper_triu)
    else:
        k = 5
        values, indices = torch.sort(IP, dim=-1, descending=True) # remove itself
        neighboring_IP = values[:, 1:(k+1)]
        reduced_mean = neighboring_IP.mean()
        reduced_std = torch.std(neighboring_IP)

    return reduced_mean, reduced_std

def inter_proxy_dist_super(proxies, super_class):
    _, IP = pairwise_distance(proxies, squared=True) # cosine similarity
    super_class_mask = super_class[:, None] == super_class # same super class
    upper_idx = torch.triu_indices(IP.size()[0], IP.size()[1], 1) # take upper triangle indices

    upper_triu = IP[upper_idx[0], upper_idx[1]]
    upper_mask = super_class_mask[upper_idx[0], upper_idx[1]]

    reduced_mean = (0.8 * upper_triu * upper_mask.float()).mean() + (upper_triu * (1 - upper_mask.float())).mean() # less panelty on with-in superclass distance
    reduced_std = torch.std(upper_triu * (1 - upper_mask.float())) # make super categories approximately evenly distributed

    return reduced_mean, reduced_std


def batch_lbl_stats(y):
    '''
        Get statistics on label distribution
        :param y: torch.Tensor of shape (N,)
        :return kk_c: count of each class of shape (C,)
    '''
    print(torch.unique(y))
    kk = torch.unique(y)
    kk_c = torch.zeros(kk.size(0))
    for kx in range(kk.size(0)):
        for jx in range(y.size(0)):
            if y[jx] == kk[kx]:
                kk_c[kx] += 1

    return kk_c


def get_wrong_indices(X, T, N=15):
    k = 1
    Y = evaluation.assign_by_euclidian_at_k(X, T, k)
    Y = torch.from_numpy(Y)
    correct = [1 if t in y[:k] else 0 for t, y in zip(T, Y)]

    wrong_ind = np.where(np.asarray(correct) == 0)[0]
    wrong_labels = T[wrong_ind]
    unique_labels, wrong_freq = torch.unique(wrong_labels, return_counts=True)
    topN_wrong_classes = unique_labels[torch.argsort(wrong_freq, descending=True)[:N]].numpy()

    return wrong_ind, topN_wrong_classes


@torch.no_grad()
def inner_product_sim(X: torch.Tensor,
                      P: torch.nn.Parameter,
                      T: torch.Tensor):

    X_copy, P_copy, T_copy = X.clone(), P.clone(), T.clone()

    X_copy = F.normalize(X_copy, dim=-1, p=2).to(P_copy.device)
    P_copy = F.normalize(P_copy, dim=-1, p=2)
    IP = torch.mm(X_copy, P_copy.T)  # inner product between X and P

    IP_gt = IP[torch.arange(X_copy.shape[0]), T_copy.long()]  # get similarities to gt-class's proxies

    return IP_gt

import scipy
def get_rho(X):
    # X = F.normalize(X, p=2, dim=-1)
    u, s, v = torch.linalg.svd(X)  # compute singular value, lower value implies lower data variance
    s = s[1:].detach().cpu().numpy()  # remove first singular value cause it is over-dominant
    s_norm = s / s.sum() # TODO: use the definition in "Revisiting Training Strategies and Generalization Performance in Deep Metric"
    uniform = np.ones(len(s)) / (len(s))
    kl = scipy.stats.entropy(uniform, s_norm)
    return kl

def get_intra_inter_dist(X, T):
    '''
        Get intra class distance/inter class distance
    '''
    X = F.normalize(X, p=2, dim=-1).detach().cpu().numpy() # L2-normalized embeddings
    unique_classes = torch.unique(T).sort()[0].detach().cpu().numpy() # get unique classes for all T
    dist_mat = np.zeros((len(unique_classes), len(unique_classes))) # distance matrix

    # Get class-specific embedding
    X_arrange_byT = []
    for cls in range(len(unique_classes)):
        indices = T == unique_classes[cls] # indices that belong to this class
        X_cls = X[indices, :]
        X_arrange_byT.append(X_cls)

    # O(C^2) to calculate inter, intra distance
    for i in range(len(unique_classes)):
        for j in range(i, len(unique_classes)):
            pairwise_dists = distance.cdist(X_arrange_byT[i], X_arrange_byT[j], 'cosine')
            avg_pairwise_dist = np.sum(pairwise_dists) / (np.prod(pairwise_dists.shape) - len(pairwise_dists.diagonal())) # take mean (ignore diagonal)
            dist_mat[i, j] = dist_mat[j, i] = avg_pairwise_dist

    # average intra-class distance
    avg_intra = dist_mat.diagonal().mean()
    # average inter-class distance
    non_diag = np.where(~np.eye(dist_mat.shape[0],dtype=bool))
    reduced_dist_mat = dist_mat[non_diag[0], non_diag[1]] # mask diagonal
    avg_inter = reduced_dist_mat.mean()
    # intra/inter ratio
    ratio = avg_intra / avg_inter
    return ratio, reduced_dist_mat, unique_classes, dist_mat


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    '''
        RFF with functions randomly drawn from \sqrt(2)cos(wx+\phi), w ~ N(0,1), \phi ~ Unif(0, 2pi)
    '''
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f))

    mid = torch.matmul(x, w.t())

    mid = mid + b
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0]
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid) + torch.sin(mid))
    else:
        Z = Z * torch.cat((torch.cos(mid), torch.sin(mid)), dim=-1)

    return Z # return the function values evaluated at x

def cov(x):
    '''
        Empirical covariance matrix of x
    '''
    n = x.shape[0]
    cov = torch.matmul(x.t(), x) / n # (sz_embed, sz_embed)
    e = torch.mean(x, dim=0).view(-1, 1)
    res = cov - torch.matmul(e, e.t())

    return res

def get_RFF_cov(X, num_functions=5):
    cfeaturecs = random_fourier_features_gpu(X, num_f=num_functions, sum=True)
    loss = torch.FloatTensor([0])
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i] # take one function
        cov1 = cov(cfeaturec) # (sz_embed, sz_embed)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix) # the Frobenius norm is on different features, deduct diagonal
    return loss


