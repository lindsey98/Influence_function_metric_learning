import os
from explaination.Confusion_Case import SampleRelabel
import torch
from Influence_function.ScalableIF_utils import grad_confusion, loss_change_train, calc_influential_func_sample
from Influence_function.IF_utils import inverse_hessian_product, calc_influential_func_orig
from Influence_function.influence_function import ScalableIF, OrigIF
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig_noisy'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub_noisy';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars_noisy'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop_noisy'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop_noisy'; config_name = 'sop'; seed = 3

    '''============ Our Influence function =================='''
    IS = ScalableIF(dataset_name, seed, loss_type, config_name, test_crop)
    basedir = 'MislabelExp_Influential_data'
    os.makedirs(basedir, exist_ok=True)

    '''Mislabelled data detection'''
    if os.path.exists("{}/{}_{}_helpful_testcls{}_SIF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0)):
        helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_SIF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0))
        harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0))
        influence_values = np.load("{}/{}_{}_influence_values_testcls{}_SIF.npy".format(basedir, IS.dataset_name, IS.loss_type, 0))
    else:
        test_features = IS.get_features()  # (N, 2048)
        confusion_class_pairs = IS.get_confusion_class_pairs()
        '''Step 1: Get theta'''
        theta_dict = IS.agg_get_theta(confusion_class_pairs[0:1], save=False)

        '''Step 2: Cache training class loss changes'''
        theta = theta_dict['theta']; theta_hat = theta_dict['theta_hat']
        l_prev, l_cur = loss_change_train(IS.model, IS.criterion, IS.dl_tr, theta, theta_hat)
        grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}

        '''Step 3: Calc influence functions'''
        influence_values = np.asarray(calc_influential_func_sample(grad_loss))
        training_sample_by_influence = influence_values.argsort()  # ascending
        IS.viz_sample(IS.dl_tr, training_sample_by_influence[:10])  # helpful
        IS.viz_sample(IS.dl_tr, training_sample_by_influence[-10:])  # harmful

        helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful
        np.save("{}/{}_{}_helpful_testcls{}_SIF".format(basedir, IS.dataset_name, IS.loss_type, 0), helpful_indices)
        np.save("{}/{}_{}_harmful_testcls{}_SIF".format(basedir, IS.dataset_name, IS.loss_type, 0), harmful_indices)
        np.save("{}/{}_{}_influence_values_testcls{}_SIF".format(basedir, IS.dataset_name, IS.loss_type, 0), influence_values)

    training_sample_by_influence = influence_values.argsort()  # ascending, harmful first
    # mislabelled indices ground-truth
    gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
    overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
    cum_overlap = np.cumsum(overlap)
    plt.scatter(range(len(overlap)), cum_overlap, label='EIF')
    pass
    # TODO climbing plot

    '''Relabelled data accuracy (only relabel harmful)'''
    # TODO climbing plot


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
        IS.viz_sample(IS.dl_tr, training_sample_by_influence[:10])  # harmful
        IS.viz_sample(IS.dl_tr, training_sample_by_influence[-10:])  # helpful

        helpful_indices = np.where(influence_values > 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values < 0)[0]  # cache all harmful
        np.save("{}/{}_{}_helpful_testcls{}_IF".format(basedir, IS.dataset_name, IS.loss_type, 0), helpful_indices)
        np.save("{}/{}_{}_harmful_testcls{}_IF".format(basedir, IS.dataset_name, IS.loss_type, 0), harmful_indices)
        np.save("{}/{}_{}_influence_values_testcls{}_IF".format(basedir, IS.dataset_name, IS.loss_type, 0), influence_values)

    training_sample_by_influence = influence_values.argsort()  # ascending, harmful first
    # TODO climbing plot
    # mislabelled indices ground-truth
    gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices
    overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
    cum_overlap = np.cumsum(overlap)
    plt.scatter(range(len(overlap)), cum_overlap, label='IF')
    plt.legend()

    '''Relabelled data accuracy (only relabel harmful)'''
    # TODO climbing plot
    overlap = np.isin(np.arange(len(IS.dl_tr.dataset)), gt_mislabelled_indices)
    cum_overlap = np.cumsum(overlap)
    plt.scatter(range(len(overlap)), cum_overlap, label='random')
    plt.legend()
    plt.show()

