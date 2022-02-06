
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
from PIL import Image
from matplotlib import cm

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
def predict_batchwise_debug(model, dataloader):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    model.eval()
    embeddings = torch.tensor([])
    labels = torch.tensor([])
    # extract batches (A becomes list of samples)
    for ct, (x, y, _) in tqdm(enumerate(dataloader)):
        # predict model output for image
        m = model(x.cuda())
        embeddings = torch.cat((embeddings, m.detach().cpu()), dim=0)
        labels = torch.cat((labels, y), dim=0)
    model.train()
    return embeddings, labels, _

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


def inter_dist(thetas, prev_thetas, cosine=True):
    thetas = thetas.reshape(len(thetas), -1)
    prev_thetas = prev_thetas.reshape(len(prev_thetas), -1)

    D, IP = pairwise_distance(
             torch.cat(
                [thetas, prev_thetas]
            ), squared=True)

    Dinter, IPinter = D[:len(thetas), :len(thetas)], IP[:len(thetas), :len(thetas)]
    D2prev, IP2prev = D[len(thetas):, len(thetas):], IP[len(thetas):, len(thetas):]

    if cosine:
        upper_triu_inter = torch.triu(IPinter, diagonal=1)
        upper_triu2prev = torch.triu(IP2prev, diagonal=1)
    else:
        upper_triu_inter = torch.triu(-Dinter, diagonal=1)
        upper_triu2prev = torch.triu(-D2prev, diagonal=1)

    reduced_mean_inter = upper_triu_inter.mean()
    reduced_std_inter = torch.std(upper_triu_inter)
    reduced_mean2prev = upper_triu2prev.mean()
    reduced_std2prev = torch.std(upper_triu2prev)

    return reduced_mean_inter, reduced_std_inter, \
           reduced_mean2prev, reduced_std2prev


def inter_proxy_dist_super(proxies, super_class):
#     proxies = F.normalize(proxies, p=2, dim=-1)
    _, IP = pairwise_distance(proxies, squared=True) # cosine similarity
    super_class_mask = super_class[:, None] == super_class # same super class

    upper_idx = torch.triu_indices(IP.size()[0], IP.size()[1], 1) # take upper triangle indices
    upper_triu = IP[upper_idx[0], upper_idx[1]]
    upper_mask = super_class_mask[upper_idx[0], upper_idx[1]]

    reduced_mean = (0.8 * upper_triu * upper_mask.float()).mean() + (upper_triu * (1 - upper_mask.float())).mean() # less panelty on with-in superclass distance
    reduced_std = 0.8 * torch.std(upper_triu * upper_mask.float()) + torch.std(upper_triu * (1 - upper_mask.float())) # make super categories approximately evenly distributed
    # reduced_std = torch.std(upper_triu * (1 - upper_mask.float())) # make super categories approximately evenly distributed
    D_within_super = upper_triu[upper_mask != 0]
    D_inter_super = upper_triu[1 - upper_mask.float() != 0]
    D_diff =  D_inter_super.mean() - D_within_super.mean()
    D_diff_clamp = 0.2 * torch.clamp(D_diff + 0.2, min=0.0) # if difference (in cosine) is larger than 0.2, all good
    return reduced_mean, reduced_std, D_diff_clamp


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



def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.7) -> (Image.Image, np.ndarray):

    """Overlay a colormapped mask on a background image

    Example::
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> from torchcam.utils import overlay_mask
        >>> img = ...
        >>> cam = ...
        >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    # Overlay the image with the mask
    img = np.asarray(img)
    if len(img.shape) < 3: # create a dummy axis if img is single channel
        img = img[:, :, np.newaxis]
    overlayed_img = Image.fromarray((alpha * img + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img, overlay

def get_wrong_indices(X, T, topk=None):
    nn_k = 1
    Y = evaluation.assign_by_euclidian_at_k(X, T, nn_k)
    Y = torch.from_numpy(Y)
    correct = [1 if t in y[:nn_k] else 0 for t, y in zip(T, Y)]

    wrong_ind = np.where(np.asarray(correct) == 0)[0] # wrong indices
    wrong_labels = T[wrong_ind] # labels at those wrong indices
    wrong_preds = Y[wrong_ind] # predictions at those wrong indices

    unique_labels, wrong_freq = torch.unique(wrong_labels, return_counts=True) # count times of being wrong
    if topk is None:
        top_wrong_classes = unique_labels[torch.argsort(wrong_freq, descending=True)].numpy() # FIXME: return all test
    else:
        top_wrong_classes = unique_labels[torch.argsort(wrong_freq, descending=True)[:topk]].numpy()

    return wrong_ind, top_wrong_classes.astype(int), wrong_labels, wrong_preds





