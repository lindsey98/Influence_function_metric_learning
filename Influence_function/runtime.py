import os
from explaination.Confusion_Case import SampleRelabel
import torch
from Influence_function.ScalableIF_utils import grad_confusion, loss_change_train, calc_influential_func_sample, loss_change_train_relabel, grad_confusion_pair
from Influence_function.IF_utils import inverse_hessian_product, calc_influential_func_orig
from Influence_function.influence_function import ScalableIF, OrigIF, MCScalableIF
from explaination.Confusion_Case import kNN_label_pred
import numpy as np
import matplotlib.pyplot as plt
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = MCScalableIF(dataset_name, seed, loss_type, config_name, test_crop)
    pair = [119, 2080]

    '''============ Our Influence function =================='''
    start_time = time.time()
    mean_deltaD_deltaL = IS.MC_estimate_pair(pair, num_thetas=1)
    influence_values = np.asarray(mean_deltaD_deltaL)
    print(time.time() - start_time)

    '''============ Original Influence function ================'''
    start_time = time.time()
    train_features = IS.get_train_features()
    test_features = IS.get_features()  # (N, 2048)
    inter_dist, v = grad_confusion_pair(IS.model, test_features, [pair[0]], [pair[1]])  # dD/dtheta
    ihvp = inverse_hessian_product(IS.model, IS.criterion, v, IS.dl_tr, scale=500, damping=0.01)
    influence_values = calc_influential_func_orig(IS=IS, train_features=train_features, inverse_hvp_prod=ihvp)
    print(time.time() - start_time)

