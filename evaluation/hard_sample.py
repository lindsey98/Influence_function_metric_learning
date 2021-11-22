import json
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    log_filename = 'cars_cars_trainval_512_0_lossProxyNCA_prob_orig_test'
    with open("{0}/{1}_recorder.json".format('log', log_filename), 'rt') as handle:
        process_recorder = json.load(handle)

    # TODO: identify samples that take long time to fit
    loss_list = [process_recorder[str(epoch)]['losses'] for epoch in range(len(process_recorder.keys()))]
    loss_list = np.stack(loss_list).T # (N, T)

    sim_list = [process_recorder[str(epoch)]['similarity'] for epoch in range(len(process_recorder.keys()))]
    sim_list = np.stack(sim_list).T # (N, T)
    # sim_list += 1 # shift distribution to (0, 2)

    for j in range(loss_list.shape[0]):
        plt.plot(range(sim_list.shape[1]), sim_list[j, :], linestyle='-', color='k', linewidth=0.5)
    plt.show()

    # # TODO: how to define a sample is hard to fit (1) rate of increasing (similarity) is slow (2) final similarity is low
    # # TODO: delete those top x% hard samples and observe the performance change
    num_epochs_needed = [np.argmax(sim_list[i] > 0.5) for i in range(len(sim_list))]
    hard2fit = np.argsort(num_epochs_needed)[::-1]
    for ind in np.where(hard2fit == sim_list.shape[1])[0]:
        hard2fit = np.delete(hard2fit, ind)
    top_hard2fit = np.asarray(hard2fit[:int(len(sim_list)*0.05)])

    plt.hist(num_epochs_needed)
    plt.show()
    plt.hist(np.asarray(num_epochs_needed)[top_hard2fit])
    plt.show()

    np.save(os.path.join('hard_samples_ind',
                         'cars_ProxyNCA_prob_orig_hard_fit.npy'), top_hard2fit)