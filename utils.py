
from __future__ import print_function
from __future__ import division

import evaluation
import numpy as np
import torch
import logging
import loss
import json
import networks
import time
#import margin_net
import similarity
import torch.nn.functional as F
from tqdm import tqdm
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
    # print(list(model.parameters())[0].device)
    model.eval()
    ds = dataloader.dataset
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
                    for c in range(criterion.nb_classes)]
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
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dataloader)
    # X = X.detach().cpu()
    # T = T.detach().cpu()
    
    print('done collecting prediction')

    #eval_time = time.time() - eval_time
    #logging.info('Eval time: %.2f' % eval_time)

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

    logging.info("NMI: {:.3f}".format(nmi * 100))

    # get predictions by assigning nearest 8 neighbors with euclidian
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

    eval_time = time.time() - eval_time
    logging.info('Eval time: %.2f' % eval_time)
    return nmi, recall


def evaluate_inshop(model, dl_query, dl_gallery,
        K = [1, 10, 20, 30, 40, 50], with_nmi = False):
    '''
        Evaluate on Inshop dataset
    '''

    # calculate embeddings with model and get targets
    X_query, T_query, *_ = predict_batchwise_inshop(
        model, dl_query)
    X_gallery, T_gallery, *_ = predict_batchwise_inshop(
        model, dl_gallery)

    nb_classes = dl_query.dataset.nb_classes()

    assert nb_classes == len(set(T_query))
    #assert nb_classes == len(T_query.unique())

    # calculate full similarity matrix, choose only first `len(X_query)` rows
    # and only last columns corresponding to the column
    T_eval = torch.cat(
        [torch.from_numpy(T_query), torch.from_numpy(T_gallery)])
    X_eval = torch.cat(
        [torch.from_numpy(X_query), torch.from_numpy(X_gallery)])
    D = similarity.pairwise_distance(X_eval)[:len(X_query), len(X_query):]

    #D = torch.from_numpy(D)
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

    return nmi, recall


@torch.no_grad()
def inner_product_sim(X: torch.Tensor, P: torch.nn.Parameter, T: torch.Tensor,
                      mask: torch.Tensor, nb_classes:int, max_proxy_per_class:int):
    '''
        get maximum inner product similarity to ground-truth proxy
        :param X: embedding torch.Tensor of shape (N, sz_embed)
        :param P: learnt proxy torch.Tensor of shape (C * self.max_proxy_per_class, sz_embed)
        :param T: one-hot ground-truth class label torch.Tensor of shape (N, C)
        :param mask: mask on activated proxies
        :param nb_classes: number of classes
        :param max_proxy_per_class: maximum proxies per class
        :return L_IP: Inner product similarity to the closest gt-class proxies
        :return cls_labels: gt-class labels
    '''
    X_copy, P_copy, T_copy = X.clone(), P.clone(), T.clone()

    X_copy = F.normalize(X_copy, dim=-1, p=2).to(P_copy.device)
    P_copy = F.normalize(P_copy, dim=-1, p=2)
    mask = mask.view(nb_classes * max_proxy_per_class, -1).to(P_copy.device)
    masked_P = P_copy * mask # mask unactivated proxies
    IP = torch.mm(X_copy, masked_P.T)  # inner product between X and P of shape (N, C*maxP)
    IP_reshape = IP.reshape((X_copy.shape[0], nb_classes, max_proxy_per_class))  # reshape to (N, C, maxP)

    IP_gt = IP_reshape[torch.arange(X_copy.shape[0]), T_copy.long(), :]  # get similarities to gt-class's proxies, of shape (N, maxP)
    rescale_IP_gt = (IP_gt+1)*(IP_gt!=0) # IP_gt range from [-1, 1]
    _, max_indices = torch.max(rescale_IP_gt, dim=-1)  # get maximum similarity to gt-class's proxies, of shape (N,)
    L_IP = IP_gt[torch.arange(IP_gt.shape[0]), max_indices]

    return L_IP.detach().cpu().numpy(), T_copy.detach().cpu().numpy()

@torch.no_grad()
def get_centers(dl_tr, model, sz_embedding):
    '''
        Compute centroid for each class
        :param dl_tr: data loader
        :param model: embedding model
        :param sz_embedding: size of embedding
        :return c_centers: class centers of shape (C, sz_embedding)
    '''
    c_centers = torch.zeros(dl_tr.dataset.nb_classes(), sz_embedding).cuda()
    n_centers = torch.zeros(dl_tr.dataset.nb_classes()).cuda()
    for ct, (x, y, _) in enumerate(dl_tr):
        with torch.no_grad():
            m = model(x.cuda())
        for ix in range(m.size(0)):
            c_centers[y] += m[ix]
            n_centers[y] += 1

    for ix in range(n_centers.size(0)):
        c_centers[ix] = c_centers[ix] / n_centers[ix]

    return c_centers


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

