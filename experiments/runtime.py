import os
from Influence_function.EIF_utils import grad_confusion_pair
from Influence_function.IF_utils import inverse_hessian_product, calc_influential_func_orig
from Influence_function.influence_function import MCScalableIF
import numpy as np
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    # dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    dataset_name = 'inshop'; config_name = 'inshop'; seed = 4

    IS = MCScalableIF(dataset_name, seed, loss_type, config_name, test_crop)
    # pairs = [[97, 4885],
    #         [109, 4837],
    #         [122, 402],
    #         [141, 411],
    #         [180, 460],
    #         [186, 2065]]
    pairs = [[5, 26418],
            [9, 16465],
            [54, 6410],
            [63, 9299],
            [68, 24062],
            [80, 804],
            [103, 1171],
            [108, 1180],
            [114, 4361],
            [119, 3087]]

    for pair in pairs:
        print(pair)
        '''============ Our Influence function =================='''
        start_time = time.time()
        mean_deltaD_deltaL = IS.MC_estimate_forpairs(pair, num_thetas=1)
        influence_values = np.asarray(mean_deltaD_deltaL)
        print('EIF runtime:', time.time() - start_time)

        '''============ Original Influence function ================'''
        start_time = time.time()
        train_features = IS.get_train_features()
        test_features = IS.get_test_features()  # (N, 2048)
        inter_dist, v = grad_confusion_pair(IS.model, test_features, [pair[0]], [pair[1]])  # dD/dtheta
        ihvp = inverse_hessian_product(IS.model, IS.criterion, v, IS.dl_tr, scale=500, damping=0.01)
        influence_values = calc_influential_func_orig(IS=IS, train_features=train_features, inverse_hvp_prod=ihvp)
        print('IF runtime:', time.time() - start_time)

