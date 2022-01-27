import os
import torch
from Influence_function.ScalableIF_utils import grad_confusion, loss_change_train_relabel
from Influence_function.IF_utils import inverse_hessian_product, calc_influential_func_orig
from Influence_function.influence_function import OrigIF, MCScalableIF
from Influence_function.Sample_relabel import kNN_label_pred
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig_noisy'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub_noisy';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars_noisy'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop_noisy'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop_noisy'; config_name = 'sop'; seed = 3

    '''============ Our Influence function =================='''
    IS = MCScalableIF(dataset_name, seed, loss_type, config_name, test_crop)
    basedir = 'MislabelExp_Influential_data'
    os.makedirs(basedir, exist_ok=True)

    for num_thetas in [1, 2, 3]:

        '''Mislabelled data detection'''
        if os.path.exists("{}/{}_{}_helpful_testcls{}_SIF_theta{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas)):
            helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_SIF_theta{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas))
            harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas))
            influence_values = np.load("{}/{}_{}_influence_values_testcls{}_SIF_theta{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas))
        else:
            confusion_class_pairs = IS.get_confusion_class_pairs()

            '''Step 1: Get deltaD_deltaL'''
            mean_deltaD_deltaL = IS.MC_estimate_group(confusion_class_pairs[0], num_thetas=num_thetas)

            '''Step 2: Calc influence functions'''
            influence_values = np.asarray(mean_deltaD_deltaL)
            training_sample_by_influence = influence_values.argsort()  # ascending
            IS.viz_samples(IS.dl_tr, training_sample_by_influence[:10])  # helpful
            IS.viz_samples(IS.dl_tr, training_sample_by_influence[-10:])  # harmful

            helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
            harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful
            np.save("{}/{}_{}_helpful_testcls{}_SIF_theta{}".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas), helpful_indices)
            np.save("{}/{}_{}_harmful_testcls{}_SIF_theta{}".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas), harmful_indices)
            np.save("{}/{}_{}_influence_values_testcls{}_SIF_theta{}".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas), influence_values)

        training_sample_by_influence = influence_values.argsort()  # ascending, harmful first
        # mislabelled indices ground-truth
        gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
        overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
        cum_overlap = np.cumsum(overlap)
        ticks_pos = np.arange(0, 1.2, 0.2)
        fraction_data_scanned = [int(x) for x in ticks_pos*len(training_sample_by_influence)]
        fraction_mislabelled_detected = [int(x) for x in ticks_pos*len(gt_mislabelled_indices)]
        plt.plot(cum_overlap, label='EIF with N_theta = {}'.format(num_thetas))
        plt.xlabel("Fraction of training data checked", fontdict={'fontsize': 15})
        plt.ylabel("Fraction of mislabelled data detected", fontdict={'fontsize': 15})
        plt.xticks(fraction_data_scanned, [round(x, 1) for x in ticks_pos])
        plt.yticks(fraction_mislabelled_detected, [round(x, 1) for x in ticks_pos])

    '''Relabelled data accuracy (only relabel harmful)'''
    # TODO climbing plot
    '''Weighted KNN'''
    harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, 1))
    relabel_dict = {}
    unique_labels, unique_counts = torch.unique(IS.train_label, return_counts=True)
    median_shots_percls = unique_counts.median().item()
    _, prob_relabel = kNN_label_pred(query_indices=harmful_indices, embeddings=IS.train_embedding,
                                     labels=IS.train_label,
                                     nb_classes=IS.dl_tr.dataset.nb_classes(), knn_k=median_shots_percls)
    for kk in range(len(harmful_indices)):
        relabel_dict[harmful_indices[kk]] = prob_relabel[kk].detach().cpu().numpy()
    # '''IF guided'''
    # relabel_dict = {}
    # theta_orig = IS.model.module[-1].weight.data
    # test_features = IS.get_features()
    # torch.cuda.empty_cache()
    # confusion_class_pairs = IS.get_confusion_class_pairs()
    # wrong_cls = confusion_class_pairs[0][0][0]
    # confused_classes = [x[1] for x in confusion_class_pairs[0]]
    # theta = IS.agg_get_theta(all_features=test_features, wrong_cls=wrong_cls, confused_classes=confused_classes)
    #
    # unique_labels, unique_counts = torch.unique(IS.train_label, return_counts=True)
    # median_shots_percls = unique_counts.median().item()
    # pred_label, _ = kNN_label_pred(query_indices=harmful_indices, embeddings=IS.train_embedding, labels=IS.train_label,
    #                                nb_classes=IS.dl_tr.dataset.nb_classes(), knn_k=median_shots_percls)
    # pred_label = pred_label[:, :5]  # top 5 relabel candidate
    # relabel_candidate = {}
    # for i, kk in enumerate(harmful_indices):
    #     relabel_candidate[kk] = pred_label[i]
    #
    # l_prev, l_cur = loss_change_train_relabel(IS.model, IS.criterion, IS.dl_tr, relabel_candidate, theta_orig, theta, harmful_indices)
    # l_diff = l_cur - l_prev  # (N_harmful, nb_classes)
    # l_diff_filtered = (l_diff < 0) * np.abs(l_diff)  # find the label when loss is decreasing -> relabeling helps to deconfuse
    # prob_relabel = l_diff_filtered / np.sum(l_diff_filtered, axis=-1, keepdims=True)
    #
    # for kk in range(len(harmful_indices)):
    #     relabel_dict[harmful_indices[kk]] = np.zeros(IS.dl_tr.dataset.nb_classes())
    #     relabel_dict[harmful_indices[kk]][pred_label[kk].long()] = prob_relabel[kk]
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


    '''============ Original Influence function =================='''
    IS = OrigIF(dataset_name, seed, loss_type, config_name, test_crop)
    basedir = 'MislabelExp_Influential_data'
    os.makedirs(basedir, exist_ok=True)

    '''Mislabelled data detection'''
    if os.path.exists("{}/{}_{}_helpful_testcls{}_IF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0)):
        helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_IF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0))
        harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_IF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0))
        influence_values = np.load("{}/{}_{}_influence_values_testcls{}_IF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0))
    else:
        train_features = IS.get_train_features()
        test_features = IS.get_features()  # (N, 2048)
        confusion_class_pairs = IS.get_confusion_class_pairs()
        wrong_cls = confusion_class_pairs[0][0][0]
        confused_classes = [x[1] for x in confusion_class_pairs[0]]

        '''Step 1: Get grad(test)'''
        inter_dist, v = grad_confusion(IS.model, test_features, wrong_cls, confused_classes,
                                       IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)  # dD/dtheta
        torch.save(v, os.path.join(basedir, 'grad_test_{}_{}_{}.pth'.format(IS.dataset_name, IS.loss_type, wrong_cls)))

        '''Step 2: Get H^-1 grad(test)'''
        ihvp = inverse_hessian_product(IS.model, IS.criterion, v, IS.dl_tr, scale=500, damping=0.01)

        '''Step 3: Get influential indices, i.e. grad(test) H^-1 grad(train), save'''
        influence_values = calc_influential_func_orig(IS=IS, train_features=train_features, inverse_hvp_prod=ihvp)
        influence_values = np.asarray(influence_values).flatten()
        training_sample_by_influence = influence_values.argsort()  # ascending
        IS.viz_samples(IS.dl_tr, training_sample_by_influence[:10])  # harmful
        IS.viz_samples(IS.dl_tr, training_sample_by_influence[-10:])  # helpful

        helpful_indices = np.where(influence_values > 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values < 0)[0]  # cache all harmful
        np.save("{}/{}_{}_helpful_testcls{}_IF".format(basedir, IS.dataset_name, IS.loss_type, 0), helpful_indices)
        np.save("{}/{}_{}_harmful_testcls{}_IF".format(basedir, IS.dataset_name, IS.loss_type, 0), harmful_indices)
        np.save("{}/{}_{}_influence_values_testcls{}_IF".format(basedir, IS.dataset_name, IS.loss_type, 0), influence_values)

    training_sample_by_influence = influence_values.argsort()  # ascending, harmful first
    # mislabelled indices ground-truth
    gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
    overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
    cum_overlap = np.cumsum(overlap)

    plt.plot(cum_overlap, label='IF')

    '''Relabelled data accuracy (only relabel harmful)'''
    overlap = np.isin(np.arange(len(IS.dl_tr.dataset)), gt_mislabelled_indices)
    cum_overlap = np.cumsum(overlap)

    plt.plot(cum_overlap, label='random')
    plt.legend()
    plt.show()

