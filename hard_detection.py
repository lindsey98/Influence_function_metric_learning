import json
import numpy as np
from sklearn.cluster import KMeans
import torch
import scipy.stats
from sklearn.metrics import pairwise_distances
from utils import predict_batchwise
from parametric_umap import *

def hard_potential(sim_dict, cls_dict, current_t, rolling_t=5, ts_sim=0.5, ts_ratio=[0.4, 1]):
    '''
        Compute the hard indices
        :param sim_dict: Dictionary for inner product similarities
        :param cls_dict: Dictionary for class labels
        :param current_t: current epoch
        :param rolling_t: number of previous rolling epochs
        :param ts_sim: threshold to determine similarity is high or not
        :param ts_ratio: lower/upper threshold to decide whether hard examples are prevalent
        :param len_training: number of training data
    '''

    update = False
    sim_prev_list = [sim_dict[str(x)] for x in range(current_t-rolling_t, current_t)]
    sim_prev = np.stack(sim_prev_list).T

    indices = np.where((sim_prev <= ts_sim).all(1) == True)[0] # get low similarity indices

    returned_indices = {}
    for cls in set(cls_dict[str(current_t-1)]): # loop over classses
        indices_cls = np.where(np.array(cls_dict[str(current_t-1)]) == cls)[0]
        num_sample_cls = len(indices_cls) # number of samples belongs to that class
        ratio_cls = np.sum(np.isin(indices_cls, indices)) / num_sample_cls

        if ratio_cls >= ts_ratio[0] and ratio_cls <= ts_ratio[1]: # if a bunch of samples (but not all) are far away from proxy
            returned_indices[cls] = indices_cls
            update = True

    return update, returned_indices

def ANOVA(embedding:np.ndarray, n_cluster:int, significance_level=0.001):
    '''
        One-way ANOVA test on H0: u1 = u2 = ...  = uk
        :param embedding: numpy.ndarray of shape (N, sz_embedding)
        :param n_cluster: k
        :param significance_level: test significance level
    '''
    assert n_cluster > 1
    significance_thr = scipy.stats.f.ppf(q=1-significance_level, dfn=(n_cluster - 1), dfd=(len(embedding) - n_cluster)) # critical value @ significance level 0.05
    clustering = KMeans(n_cluster).fit(embedding)
    WSS = clustering.inertia_ # within-SS
    TSS = np.sum((embedding - embedding.mean(0)) ** 2) # total-SS
    BSS = TSS - WSS # between-SS
    MSB = BSS / (n_cluster - 1) # mean-BSS
    MSW = WSS / (len(embedding) - n_cluster) # mean-WSS
    if MSW == 0:
        return 0, significance_thr, False
    F = MSB / MSW # F-statistics
    return F, significance_thr, F > significance_thr

def FullReduceTest(embedding:np.ndarray, n_cluster:int, significance_level=0.001):
    '''
        Full Reduced Model Test H0: k-1 centers is sufficient, H1: need to have k centers
        :param embedding: numpy.ndarray of shape (N, sz_embedding)
        :param n_cluster: k
        :param significance_level: test significance level
    '''
    assert n_cluster > 1
    N = len(embedding)
    k = n_cluster
    significance_thr = scipy.stats.f.ppf(q=1-significance_level,
                                         dfn=1, dfd=(N-k)) # critical value @ significance level 0.05
    # full model has k groups
    clustering = KMeans(k).fit(embedding)
    WSS = clustering.inertia_  # within-SS or SSE
    MSW = WSS / (N-k)  # mean-WSS
    if MSW == 0: # if zero within-SS, duplicate data and perfect clustering
        return 0.0, significance_thr, False

    # reduced model has k-1 groups
    if k-1 == 1:
        WSS_old = np.sum((embedding - embedding.mean(0)) ** 2) # if only 1 cluster, then WSS = TSS
    else:
        clustering = KMeans(k-1).fit(embedding)
        WSS_old = clustering.inertia_  # within-SS or SSE

    F = ((WSS_old - WSS) / 1) / MSW
    return F, significance_thr, F > significance_thr

def split_potential(embeddings:np.ndarray, labels:np.ndarray, no_proxies:np.ndarray, significance_level=0.001):
    '''
        Compute cluster spliting potential by significance test on number of proxies
        :param embeddings: of shape (N, sz_embedding)
        :param labels: of shape (N,)
        :param no_proxies: of shape (C,)
    '''
    returned_indices = {}
    update = False

    for cls in set(labels.tolist()):
        indices_cls = np.where(np.array(labels == cls))[0]
        embedding_cls = embeddings[indices_cls]
        count_proxy = int(no_proxies[cls])
        _, _, significant = FullReduceTest(embedding_cls, count_proxy + 1, significance_level)
        if significant:
            update = True
            returned_indices[cls] = indices_cls

    return update, returned_indices

if __name__ == '__main__':

    data_name = 'logo2k'
    # with open('./log/{}_{}_trainval_2048_0_True_ip.json'.format(data_name, data_name), 'rt') as handle:
    #     sim_dict = json.load(handle)
    # with open('./log/{}_{}_trainval_2048_0_True_cls.json'.format(data_name, data_name), 'rt') as handle:
    #     cls_dict = json.load(handle)

    # gt_hard = [3, 20, 36, 39, 40, 41, 47, 49, 52, 53, 56, 62, 65, 68, 71, 73, 76, 78, 83, 84, 89, 96, 99,
    #           106, 108, 110, 114, 118, 124, 127, 128, 129, 131, 138, 139, 141, 142, 144, 148, 152, 158, 162, 172, 181, 184, 191, 192, 193, 195, 196, 199,
    #            ]
    # print(len(gt_hard))

    # for t in range(30, 31):
    #     update, indices = hard_potential(sim_dict=sim_dict, cls_dict=cls_dict,
    #                                      current_t=t, rolling_t=5, ts_sim=0.5)
    #     print("Epoch {}, update is {}, number of classes need to update is {}".format(t, update, len(indices.keys())))
    #     print(indices.keys())
    #     # print(len(set(indices.keys()).intersection(set(gt_hard))) / (len(set(gt_hard))))
    #     print()
    #     # print(indices)
    dl_tr, dl_ev = prepare_data(data_name='logo2k', root='dvi_data_{}_{}/'.format('logo2k', 'True'), save=False)

    # load model
    feat = Feat_resnet50_max_n()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, 2048)
    model = torch.nn.Sequential(feat, emb)
    model = nn.DataParallel(model)
    model.cuda()

    for t in range(20, 21):

        model_dir = 'dvi_data_{}_{}_Ftest/ResNet_2048_Model'.format('logo2k', 'True')
        model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_2048_0.pth'.format(model_dir, t, 'logo2k', 'logo2k')))

        # embedding, label, *_ = predict_batchwise(model, dl_tr)
        # torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(model_dir, t))
        # torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(model_dir, t))

        embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(model_dir, t)).detach().cpu().numpy()
        label = torch.load('{}/Epoch_{}/training_labels.pth'.format(model_dir, t)).detach().cpu().numpy()
        mask = torch.load('{}/Epoch_{}/proxy.pth'.format(model_dir, t), map_location='cpu')['mask'].detach()
        count_proxy = torch.sum(mask, -1).detach().cpu().numpy()
        print(len(set(label.tolist())))
        # F, thr, test_result = ANOVA(embedding_sub, n_cluster=2)
        # print(F, thr, test_result)
        # embedding_sub = embedding[label == 0]
        # print(ANOVA(embedding_sub, n_cluster=2))
        # print(FullReduceTest(embedding_sub, n_cluster=3))
        # print(clustering_tradeoff(embedding_sub, n_cluster=2))
        # print(clustering_tradeoff(embedding_sub, n_cluster=3))
        # print(clustering_tradeoff(embedding_sub, n_cluster=4))
        update, indices = split_potential(embedding, label, count_proxy, 0.005)
        print("Epoch {}, update is {}, number of classes need to update is {}".format(t, update, len(indices.keys())))
        print(indices.keys())