from Influence_function.influence_function import OrigIF, EIF
from Influence_function.EIF_utils import grad_confusion
import os
from Influence_function.IF_utils import *
import numpy as np
from evaluation import assign_by_euclidian_at_k_indices

os.environ['CUDA_VISIBLE_DEVICES'] = "1, 0"

if __name__ == '__main__':
    sz_embedding = 512; epoch = 40; test_crop = False; topk_cls = 30
    loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cub';  config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cars'; config_name = 'cars_ProxyNCA_prob_orig'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'inshop'; config_name = 'inshop_ProxyNCA_prob_orig'; seed = 4

    # loss_type = 'SoftTriple'; dataset_name = 'cub'; config_name = 'cub_SoftTriple'; seed = 3
    # loss_type = 'SoftTriple'; dataset_name = 'cars'; config_name = 'cars_SoftTriple'; seed = 4
    # loss_type = 'SoftTriple'; dataset_name = 'inshop'; config_name = 'inshop_SoftTriple'; seed = 3

    IS_IF = OrigIF(dataset_name, seed, loss_type, config_name, 'dataset/config.json', test_crop, sz_embedding, epoch, 'ResNet')
    IS_EIF = EIF(dataset_name, seed, loss_type, config_name, 'dataset/config.json', test_crop, sz_embedding, epoch, 'ResNet')

    '''Given a confusion pair, find its top 5 harmful training identified by EIF and IF'''
    '''Step 1: Get all wrong pairs'''
    testing_embedding, testing_label = IS_IF.testing_embedding, IS_IF.testing_label
    test_nn_indices, test_nn_label = assign_by_euclidian_at_k_indices(testing_embedding, testing_label, 1)
    wrong_indices = (test_nn_label.flatten() != testing_label.detach().cpu().numpy().flatten()).nonzero()[0]
    confuse_indices = test_nn_indices.flatten()[wrong_indices]
    print(len(confuse_indices))
    assert len(wrong_indices) == len(confuse_indices)

    train_features = IS_IF.get_train_features()
    test_features = IS_IF.get_test_features()  # (N, 2048)
    for kk in range(min(len(wrong_indices), 100)):
        wrong_ind = wrong_indices[kk]
        confuse_ind = confuse_indices[kk]

        # IF
        influence_values_IF = IS_IF.influence_func_forpairs(train_features=train_features, test_features=test_features,
                                                            wrong_indices=[wrong_ind], confuse_indices=[confuse_ind])
        influence_values_IF = np.asarray(influence_values_IF).flatten()
        top_5_harmful_indices_IF = influence_values_IF.argsort()

        # EIF
        mean_deltaL_deltaD_EIF = IS_EIF.MC_estimate_forpair([wrong_ind, confuse_ind], num_thetas=1, steps=50)
        influence_values_EIF = np.asarray(mean_deltaL_deltaD_EIF)
        top_5_harmful_indices_EIF = influence_values_EIF.argsort()[::-1]

        print(top_5_harmful_indices_IF)
        print(top_5_harmful_indices_EIF)
        break