
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

def predict_batchwise_loss(model, dataloader, criterion):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    model.eval()
    indices = torch.tensor([])
    normalize_prob = torch.tensor([])
    base_loss = torch.tensor([])
    embeddings = torch.tensor([])
    labels = torch.tensor([])
    gt_D_weighted = torch.tensor([])

    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for (x, y, indices) in tqdm(dataloader, desc="Batch-wise prediction"):
            x, y = x.to(list(model.parameters())[0].device), y.to(list(model.parameters())[0].device)
            m = model(x)
            indices_bth, normalize_prob_bth, gt_D_weighted_bth, base_loss_bth, Proxy_IP = criterion.loss4debug(m, indices, y)
            indices = torch.cat((indices, indices_bth.detach().cpu()), dim=0)
            normalize_prob = torch.cat((normalize_prob, normalize_prob_bth.detach().cpu()), dim=0)
            base_loss = torch.cat((base_loss, base_loss_bth.detach().cpu()), dim=0)
            embeddings = torch.cat((embeddings, m.detach().cpu()), dim=0)
            labels = torch.cat((labels, y.detach().cpu()), dim=0)
            gt_D_weighted = torch.cat((gt_D_weighted, gt_D_weighted_bth.detach().cpu()), dim=0)

    # compute proxy2proxy similarity
    _blk_mask = [torch.ones((criterion.max_proxy_per_class, criterion.max_proxy_per_class)) \
                    for _ in range(criterion.nb_classes)]
    blk_mask = torch.block_diag(*_blk_mask).to(Proxy_IP.device)
    p2p_sim = Proxy_IP * blk_mask

    return embeddings, labels, indices, normalize_prob, gt_D_weighted, base_loss, p2p_sim

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

def inter_proxy_dist(proxies, cosine=True):
    D, IP = pairwise_distance(proxies, squared=True)
    if cosine:
        upper_triu = torch.triu(IP, diagonal=1)
    else:
        upper_triu = torch.triu(D, diagonal=1)
    reduced_mean = upper_triu.mean()
    reduced_var = torch.std(upper_triu)
    return reduced_mean, reduced_var

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


