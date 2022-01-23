from .influence_function import *


if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    # dataset_name = 'cub';  config_name = 'cub'; seed = 0
    dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = OrigIF(dataset_name, seed, loss_type, config_name, test_crop)