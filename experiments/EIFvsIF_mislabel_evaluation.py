import os
import torch
from Influence_function.EIF_utils import grad_confusion
from Influence_function.IF_utils import inverse_hessian_product, calc_influential_func_orig
from Influence_function.influence_function import OrigIF, EIF
from Influence_function.sample_relabel import kNN_label_pred
import numpy as np
import matplotlib.pyplot as plt
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    noisy_level = 0.1
    sz_embedding = 512; epoch = 40; test_crop = False
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'cub_noisy';  config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'cars_noisy'; config_name = 'cars_ProxyNCA_prob_orig'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'inshop_noisy'; config_name = 'inshop_ProxyNCA_prob_orig'; seed = 4

    # loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'cub_noisy';  config_name = 'cub_SoftTriple'; seed = 3
    # loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'cars_noisy';  config_name = 'cars_SoftTriple'; seed = 4
    loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'inshop_noisy';  config_name = 'inshop_SoftTriple'; seed = 3

    '''============================================= Our Empirical Influence function =============================================================='''
    IS = EIF(dataset_name, seed, loss_type, config_name, test_crop)
    basedir = 'MislabelExp_Influential_data'
    os.makedirs(basedir, exist_ok=True)

    for num_thetas in [1, 2, 3]:

        '''Mislabelled data detection'''
        if os.path.exists("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level)):
            helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
            harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
            influence_values = np.load("{}/{}_{}_influence_values_testcls{}_SIF_theta{}_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
        else:
            confusion_class_pairs = IS.get_confusion_class_pairs()

            '''Step 1: Get deltaD_deltaL'''
            mean_deltaL_deltaD = IS.MC_estimate_forclasses(confusion_class_pairs[0], num_thetas=num_thetas, steps=50)

            '''Step 2: Calc influence functions'''
            influence_values = np.asarray(mean_deltaL_deltaD)
            helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
            harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful

            np.save("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), helpful_indices)
            np.save("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), harmful_indices)
            np.save("{}/{}_{}_influence_values_testcls{}_SIF_theta{}_{}".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), influence_values)

        # mislabelled indices ground-truth
        training_sample_by_influence = np.abs(influence_values).argsort()[::-1]  # fixme: descending
        # training_sample_by_influence = influence_values.argsort()[::-1]  # fixme: harmful is positive

        gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
        overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
        cum_overlap = np.cumsum(overlap)
        ticks_pos = np.arange(0, 1.2, 0.2)
        fraction_data_scanned = [int(x) for x in ticks_pos*len(training_sample_by_influence)]
        fraction_mislabelled_detected = [int(x) for x in ticks_pos*len(gt_mislabelled_indices)]
        plt.plot(cum_overlap, label='EIF with $N_theta$ = {}'.format(num_thetas))
        plt.xlabel("Fraction of training data checked", fontdict={'fontsize': 15})
        plt.ylabel("Fraction of mislabelled data detected", fontdict={'fontsize': 15})
        plt.xticks(fraction_data_scanned, [round(x, 1) for x in ticks_pos])
        plt.yticks(fraction_mislabelled_detected, [round(x, 1) for x in ticks_pos])

    '''Relabelled data accuracy (only relabel harmful)'''
    # TODO climbing plot
    '''Weighted KNN'''
    start_time = time.time()
    harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, 1, noisy_level))
    relabel_dict = {}
    unique_labels, unique_counts = torch.unique(IS.train_label, return_counts=True)
    median_shots_percls = unique_counts.median().item()
    _, prob_relabel = kNN_label_pred(query_indices=harmful_indices, embeddings=IS.train_embedding,
                                     labels=IS.train_label,
                                     nb_classes=IS.dl_tr.dataset.nb_classes(), knn_k=median_shots_percls)
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
    print(ct_correct, total_ct)

    '''======================================================================================================================================='''

    '''======================================== Original Influence function =========================================================================='''
    IS = OrigIF(dataset_name, seed, loss_type, config_name, test_crop)
    basedir = 'MislabelExp_Influential_data'
    os.makedirs(basedir, exist_ok=True)

    '''Mislabelled data detection'''
    if os.path.exists("{}/{}_{}_helpful_testcls{}_IF_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, noisy_level)):
        helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_IF_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, noisy_level))
        harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_IF_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, noisy_level))
        influence_values = np.load("{}/{}_{}_influence_values_testcls{}_IF_{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, noisy_level))
    else:
        train_features = IS.get_train_features()
        test_features = IS.get_test_features()  # (N, 2048)
        confusion_class_pairs = IS.get_confusion_class_pairs()
        wrong_cls = confusion_class_pairs[0][0][0]
        confused_classes = [x[1] for x in confusion_class_pairs[0]]

        '''Step 1: Get grad(test)'''
        inter_dist, v = grad_confusion(IS.model, test_features, wrong_cls, confused_classes,
                                       IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)  # dD/dtheta

        '''Step 2: Get H^-1 grad(test)'''
        ihvp = inverse_hessian_product(IS.model, IS.criterion, v, IS.dl_tr, scale=500, damping=0.01)

        '''Step 3: Get influential indices, i.e. grad(test) H^-1 grad(train), save'''
        influence_values = calc_influential_func_orig(IS=IS, train_features=train_features, inverse_hvp_prod=ihvp)
        influence_values = np.asarray(influence_values).flatten()
        helpful_indices = np.where(influence_values > 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values < 0)[0]  # cache all harmful

        np.save("{}/{}_{}_helpful_testcls{}_IF_{}".format(basedir, IS.dataset_name, IS.loss_type, 0, noisy_level), helpful_indices)
        np.save("{}/{}_{}_harmful_testcls{}_IF_{}".format(basedir, IS.dataset_name, IS.loss_type, 0, noisy_level), harmful_indices)
        np.save("{}/{}_{}_influence_values_testcls{}_IF_{}".format(basedir, IS.dataset_name, IS.loss_type, 0, noisy_level), influence_values)

    # mislabelled indices ground-truth
    training_sample_by_influence = np.abs(influence_values).argsort()[::-1]  # fixme: descending
    # training_sample_by_influence = influence_values.argsort()  # fixme: harmful is negative

    gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
    overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
    cum_overlap = np.cumsum(overlap)

    plt.plot(cum_overlap, label='IF')

    '''======================================================================================================================================='''

    '''=============================================Random================================================================'''
    overlap = np.isin(np.arange(len(IS.dl_tr.dataset)), gt_mislabelled_indices)
    cum_overlap = np.cumsum(overlap)

    plt.plot(cum_overlap, label='random')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig('./images/mislabel_{}_{}_alltheta_noisylevel{}_abs.pdf'.format(dataset_name, loss_type, noisy_level),
    #             bbox_inches='tight')



