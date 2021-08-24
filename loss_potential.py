import json

import numpy as np


def loss_potential(sim_dict, cls_dict, current_t, rolling_t=5, ts_sim=0.5, ts_ratio=[0.3, 0.7]):
    '''
    Compute the hard indices
    :param sim_dict: Dictionary for inner product similarities
    :param cls_dict: Dictionary for class labels
    :param current_t: current epoch
    :param rolling_t: number of previous rolling epochs
    :param ts_diff: threshold to decide rolling difference is large or not
    :param ts_sim: threshold to determine similarity is high or not
    :param ts_ratio: lower/upper threshold to decide whether hard examples are prevalent
    :param len_training: number of training data
    '''
    # TODO: should get potential for each class
    update = False
    sim_prev = np.stack([sim_dict[str(x)] for x in range(current_t-rolling_t+1, current_t+1)]).T
    assert sim_prev.shape[1] == min(rolling_t, current_t)

    indices = np.where((sim_prev <= ts_sim).all(1) == True)[0] # get low similarity indices

    # rolling_diff = np.diff(sim_prev)
    # indices2 = np.where((np.abs(rolling_diff) <= ts_diff).all(1) == True)[0] # get slow convergence indices

    # get intersection between indices1 and indices2
    # indices = set(indices1).intersection(set(indices2))
    returned_indices = {}
    for cls in set(cls_dict[str(current_t)]): # loop over classses
        num_sample_cls = np,sum(np.array(cls_dict[str(current_t)]) == cls) # number of samples belongs to that class
        indices_cls = np.where(np.array(cls_dict[str(current_t)]) == cls)[0]
        ratio_cls = np.sum(indices_cls.isin(indices)) / num_sample_cls
        if ratio_cls >= ts_ratio[0] and ratio_cls <= ts_ratio[1]: # if a bunch of samples (but not all) are far away from proxy
            returned_indices[cls] = indices_cls
            update = True

    return update, returned_indices



if __name__ == '__main__':
    with open('./log/logo2k_logo2k_trainval_2048_0_sim.json', 'rt') as handle:
        sim_dict = json.load(handle)
    with open('./log/logo2k_logo2k_trainval_2048_0_cls.json', 'rt') as handle:
        cls_dict = json.load(handle)

    update, indices = loss_potential(sim_dict=sim_dict, cls_dict=cls_dict, current_t=5)