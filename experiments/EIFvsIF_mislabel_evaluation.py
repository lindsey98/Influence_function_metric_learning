import os
import torch
from Influence_function.EIF_utils import grad_confusion, calc_loss_train
from Influence_function.IF_utils import inverse_hessian_product, calc_influential_func_orig, grad_loss
from Influence_function.influence_function import OrigIF, EIF, kNN_label_pred
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('TkAgg')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == '__main__':
    noisy_level = 0.01
    sz_embedding = 512; epoch = 40; test_crop = False
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'cub_noisy';  config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'cars_noisy'; config_name = 'cars_ProxyNCA_prob_orig'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'inshop_noisy'; config_name = 'inshop_ProxyNCA_prob_orig'; seed = 4

    # loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'cub_noisy';  config_name = 'cub_SoftTriple'; seed = 3
    loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'cars_noisy';  config_name = 'cars_SoftTriple'; seed = 4
    # loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'inshop_noisy';  config_name = 'inshop_SoftTriple'; seed = 3

    '''============================================= Our Empirical Influence function =============================================================='''
    IS = EIF(dataset_name, seed, loss_type, config_name, 'dataset/config.json', test_crop, sz_embedding, epoch, 'ResNet', noisy_level)
    basedir = 'MislabelExp_Influential_data'
    os.makedirs(basedir, exist_ok=True)

    train_features = IS.get_train_features()
    training_loss_grad = grad_loss(IS.model, IS.criterion, train_features, IS.train_label)
    num_thetas = 1
    '''Mislabelled data detection'''
    if os.path.exists("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level)):
        helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
        harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
        influence_values = np.load("{}/{}_{}_influence_values_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
    else:
        confusion_class_pairs = IS.get_confusion_class_pairs()

        '''Step 1: Get deltaD_deltaL with the confusion pairs for the top1 frequently wrong testing class'''
        mean_deltaL_deltaD = IS.EIF_for_groups_confusion(confusion_class_pairs[0],
                                                         num_thetas=num_thetas,
                                                         steps=50)
        '''Step 2: Calc influence functions'''
        influence_values = np.asarray(mean_deltaL_deltaD)
        helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful

        np.save("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}_step50".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), helpful_indices)
        np.save("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}_step50".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), harmful_indices)
        np.save("{}/{}_{}_influence_values_testcls{}_SIF_theta{}_{}_step50".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), influence_values)

    # mislabelled indices ground-truth
    training_sample_by_influence = np.abs(influence_values).argsort()[::-1]  # fixme: descending
    training_sample_rank = np.empty_like(training_sample_by_influence)
    training_sample_rank[training_sample_by_influence] = np.arange(len(influence_values))

    # '''Relabelled data accuracy (only relabel harmful)'''
    # '''Weighted KNN'''
    gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
    start_time = time.time()
    harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name,
                                                                                            IS.loss_type, 0, num_thetas,
                                                                                            noisy_level))
    relabel_dict = {}
    unique_labels, unique_counts = torch.unique(IS.train_label, return_counts=True)
    median_shots_percls = unique_counts.median().item()
    _, prob_relabel = kNN_label_pred(query_indices=harmful_indices,
                                     embeddings=IS.train_embedding,
                                     labels=IS.train_label,
                                     nb_classes=IS.dl_tr.dataset.nb_classes(),
                                     knn_k=median_shots_percls)
    for kk in range(len(harmful_indices)):
        relabel_dict[harmful_indices[kk]] = prob_relabel[kk].detach().cpu().numpy()
    print(time.time() - start_time)

    total_ct = 0
    ct_correct = 0
    for ind in gt_mislabelled_indices:
        if ind in relabel_dict.keys():
            total_ct += 1
            if relabel_dict[ind].argsort()[::-1][0] == IS.dl_tr.dataset.ys[ind]:
                if relabel_dict[ind].argsort()[::-1][1] == IS.dl_tr_clean.dataset.ys[ind]:
                    ct_correct += 1
            else:
                if relabel_dict[ind].argsort()[::-1][0] == IS.dl_tr_clean.dataset.ys[ind]:
                    ct_correct += 1
    print(ct_correct, total_ct, ct_correct/total_ct)

