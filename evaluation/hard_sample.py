import json
import numpy as np
import matplotlib.pyplot as plt
import os

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
    log_filename = 'cub_cub_trainval_512_0_lossProxyNCA_prob_orig_test'
    with open("{0}/{1}_recorder.json".format('log', log_filename), 'rt') as handle:
        process_recorder = json.load(handle)

    # TODO: identify samples that take long time to fit
    loss_list = [process_recorder[str(epoch)]['losses'] for epoch in range(len(process_recorder.keys()))]
    loss_list = np.stack(loss_list).T # (N, T)

    sim_list = [process_recorder[str(epoch)]['similarity'] for epoch in range(len(process_recorder.keys()))]
    sim_list = np.stack(sim_list).T # (N, T)

    # for j in range(loss_list.shape[0]):
    #     plt.plot(range(sim_list.shape[1]), sim_list[j, :], linestyle='-', color='k', linewidth=0.5)
    # plt.show()

    # TODO: how to define a sample is hard to fit (1) rate of increasing (similarity) is slow (2) final similarity is low
    increasing_rate = (sim_list[:, -1] - sim_list[:, 0]) / np.abs(sim_list[:, 0]) # (final_sim-initial_sim)/initial_sim
    final_sim = sim_list[:, -1]

    # TODO: delete those top x% hard samples and observe the performance change
    bottom_increasing_rate = np.argsort(increasing_rate)[:int(len(increasing_rate)*0.05)]
    bottom_final_sim = np.argsort(final_sim)[:int(len(final_sim)*0.05)]
    np.save(os.path.join('hard_samples_ind', 'cub_ProxyNCA_prob_orig_hard_increase.npy'), bottom_increasing_rate)
    np.save(os.path.join('hard_samples_ind', 'cub_ProxyNCA_prob_orig_hard_final.npy'), bottom_final_sim)