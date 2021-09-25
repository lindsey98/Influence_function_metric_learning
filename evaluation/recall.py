
import numpy as np
import sklearn.metrics.pairwise
import torch.nn.functional as F
import torch
import math
from tqdm import tqdm

def assign_by_euclidian_at_k(X, T, k):
    """ 
        X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
        k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)
    chunk_size = 1000
    num_chunks = math.ceil(len(X)/chunk_size)
    distances = torch.tensor([])
    for i in tqdm(range(0, num_chunks)):
        chunk_indices = [chunk_size*i, min(len(X), chunk_size*(i+1))]
        chunk_X = X[chunk_indices[0]:chunk_indices[1], :]
        distance_mat = torch.from_numpy(sklearn.metrics.pairwise.pairwise_distances(X, chunk_X))
        distances = torch.cat((distances, distance_mat), dim=-1)
    assert distances.shape[0] == len(X)
    assert distances.shape[1] == len(X)

    distances = distances.numpy()
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1]

    return np.array([[T[i] for i in ii] for ii in indices])


def assign_by_euclidian_at_k_indices(X, T, k):
    """
        X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
        k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)
    chunk_size = 1000
    num_chunks = math.ceil(len(X)/chunk_size)
    distances = torch.tensor([])
    for i in tqdm(range(0, num_chunks)):
        chunk_indices = [chunk_size*i, min(len(X), chunk_size*(i+1))]
        chunk_X = X[chunk_indices[0]:chunk_indices[1], :]
        distance_mat = torch.from_numpy(sklearn.metrics.pairwise.pairwise_distances(X, chunk_X))
        distances = torch.cat((distances, distance_mat), dim=-1)
    assert distances.shape[0] == len(X)
    assert distances.shape[1] == len(X)

    distances = distances.numpy()
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1]

    return indices, np.array([[T[i] for i in ii] for ii in indices])


def calc_recall_at_k(T, Y, k):
    """
        Check whether a sample's KNN contain any sample with the same class labels as itself
        T : [nb_samples] (target labels)
        Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))


