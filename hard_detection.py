import json
import numpy as np

def hard_potential(sim_dict, cls_dict, current_t, rolling_t=5, ts_sim=0.5, ts_ratio=[0.4, 1]):
    '''
    Compute the hard indices
    :param sim_dict: Dictionary for inner product similarities
    :param cls_dict: Dictionary for class labels
    :param current_t: current epoch
    :param rolling_t: number of previous rolling epochs
    :param ts_sim: threshold to determine similarity is high or not
    :param ts_ratio: lower/upper threshold to decide whether hard examples are prevalent
    :param len_training: number of training data
    '''

    update = False
    sim_prev_list = [sim_dict[str(x)] for x in range(current_t-rolling_t, current_t)]
    sim_prev = np.stack(sim_prev_list).T

    indices = np.where((sim_prev <= ts_sim).all(1) == True)[0] # get low similarity indices

    returned_indices = {}
    for cls in set(cls_dict[str(current_t-1)]): # loop over classses
        indices_cls = np.where(np.array(cls_dict[str(current_t-1)]) == cls)[0]
        num_sample_cls = len(indices_cls) # number of samples belongs to that class
        ratio_cls = np.sum(np.isin(indices_cls, indices)) / num_sample_cls

        if ratio_cls >= ts_ratio[0] and ratio_cls <= ts_ratio[1]: # if a bunch of samples (but not all) are far away from proxy
            returned_indices[cls] = indices_cls
            update = True

    return update, returned_indices


if __name__ == '__main__':
    data_name = 'logo2k'
    with open('./log/{}_{}_trainval_2048_0_True_ip.json'.format(data_name, data_name), 'rt') as handle:
        sim_dict = json.load(handle)
    with open('./log/{}_{}_trainval_2048_0_True_cls.json'.format(data_name, data_name), 'rt') as handle:
        cls_dict = json.load(handle)

    # gt_hard = [3, 20, 36, 39, 40, 41, 47, 49, 52, 53, 56, 62, 65, 68, 71, 73, 76, 78, 83, 84, 89, 96, 99,
    #           106, 108, 110, 114, 118, 124, 127, 128, 129, 131, 138, 139, 141, 142, 144, 148, 152, 158, 162, 172, 181, 184, 191, 192, 193, 195, 196, 199,
    #            ]
    # print(len(gt_hard))

    for t in range(30, 31):
        update, indices = hard_potential(sim_dict=sim_dict, cls_dict=cls_dict,
                                         current_t=t, rolling_t=5, ts_sim=0.5)
        print("Epoch {}, update is {}, number of classes need to update is {}".format(t, update,
                                                                                      len(indices.keys())))
        print(indices.keys())
        # print(len(set(indices.keys()).intersection(set(gt_hard))) / (len(set(gt_hard))))
        print()
        # print(indices)