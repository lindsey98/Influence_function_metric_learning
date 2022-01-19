#! /usr/bin/env python3
# Code is modified from https://github.com/kohpangwei/influence-release and https://github.com/nimarb/pytorch_influence_functions
import os
import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from Influence_function.influence_function_orig import *
import pickle

def calc_influential_func(model, phi_test, inverse_hvp_prod):

    theta_L = model.module[-1].weight.data # (2048, sz_embed)
    influence_values = []
    for i in range(len(inverse_hvp_prod)):
        second_term = (-1) * torch.matmul(inverse_hvp_prod[i].flatten().detach().cpu(), phi_test.flatten())
        first_term = torch.matmul(theta_L, phi_test.t()) # FIXME
        influence_values.append(first_term + second_term)
    return influence_values

if __name__ == '__main__':
    from Influence_function.influential_sample import InfluentialSample
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)
    '''Step 1: Get grad(train)'''
    IS.model.eval()
    # # Forward propogate up to projection layer, cache the features (testing loader)
    # train_features = torch.tensor([])  # (N, 2048)
    # for ct, (x, t, _) in tqdm(enumerate(IS.dl_tr)):
    #     x = x.cuda()
    #     m = IS.model.module[:-1](x)
    #     train_features = torch.cat((train_features, m.detach().cpu()), dim=0)
    # grad_train = grad_loss(IS.model, IS.criterion, train_features, IS.train_label)
    # os.makedirs('Influential_data_baselines', exist_ok=True)
    # with open(os.path.join('Influential_data_baselines', '{}_grad4train.pkl'.format(IS.dataset_name)), 'wb') as handle:
    #     pickle.dump(grad_train, handle)

    '''Step 2: Get H^-1 v, where v = grad(train)'''
    with open(os.path.join('Influential_data_baselines', '{}_grad4train.pkl'.format(IS.dataset_name)), 'rb') as handle:
        grad_train = pickle.load(handle)
    ivp_train = []
    for it in range(len(grad_train)):
        ivp_this_train = inverse_hessian_product(model=IS.model, criterion=IS.criterion, v=grad_train[it], dl_tr=IS.dl_tr,
                            scale=500, damping=0.01)
        ivp_train.append(ivp_this_train)
    pass
    '''Step 3: Multiply grad(test) H^-1 v, where v = grad(train)'''
    '''Step 4: Finding mislabelled'''
    pass