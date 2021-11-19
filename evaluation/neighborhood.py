from utils import predict_batchwise
import torch
import similarity
import torch.nn.functional as F
import time
import evaluation
import logging
import math

def neighboring_emb_finding(X_train, X_test, K=50, length_scale=1.):

    D = torch.square(torch.cdist(X_test, X_train, p=2))

    # get top k neighbors in training set with smallest distance
    neighboring_D, neighboring_ind = D.sort(dim=-1)
    neighboring_D = neighboring_D[:, :K]
    neighboring_ind = neighboring_ind[:, :K]
    neighboring_emb = X_train[neighboring_ind] # (N_test, K, sz_embed)
    RBF_K = torch.exp(-neighboring_D / (2* length_scale**2)) # RBF kernel (N_test, K)
    RBF_K = F.normalize(RBF_K, p=1, dim=-1)

    neighboring_weighted_emb = RBF_K.unsqueeze(-1) * neighboring_emb # (N_test, K, sz_embed)
    neighboring_weighted_emb = neighboring_weighted_emb.sum(1) # (N_test, sz_embed)

    return neighboring_weighted_emb


def evaluate_neighborhood(model, X_train, X_test, T_test, length_scale, weight, recall_list=[1, 2, 4, 8]):
    # need to do L2 normalize
    X_train_normalize = F.normalize(X_train, p=2, dim=-1)
    X_test_normalize = F.normalize(X_test, p=2, dim=-1)
    X_test_neighbor = neighboring_emb_finding(X_train_normalize, X_test_normalize, K=2000, length_scale=length_scale)

    # skip connection
    X_enriched = X_test + weight * X_test_neighbor

    # Recall get predictions by assigning nearest 8 neighbors with euclidian
    max_dist = max(recall_list)
    Y = evaluation.assign_by_euclidian_at_k(X_enriched, T_test, max_dist)
    Y = torch.from_numpy(Y)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in recall_list:
        r_at_k = evaluation.calc_recall_at_k(T_test, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))


