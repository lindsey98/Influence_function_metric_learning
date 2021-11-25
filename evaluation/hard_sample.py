import json
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    log_filename = 'cub_cub_pfix_trainval_512_0_lossProxyNCA_pfix_test'
    with open("{0}/{1}_recorder.json".format('log', log_filename), 'rt') as handle:
        process_recorder = json.load(handle)

    # TODO: identify samples that take long time to fit
    loss_list = [process_recorder[str(epoch)]['losses'] for epoch in range(len(process_recorder.keys()))]
    loss_list = np.stack(loss_list).T # (N, T)

    sim_list = [process_recorder[str(epoch)]['similarity'] for epoch in range(len(process_recorder.keys()))]
    sim_list = np.stack(sim_list).T # (N, T)

    for j in range(loss_list.shape[0]):
        plt.plot(range(sim_list.shape[1]), sim_list[j, :], linestyle='-', color='k', linewidth=0.5)
    plt.show()

    # # how to define a sample is hard to fit: look at the training efforts needed in order to reach a similarity threshold (0.5)
    # # delete those top x% hard samples and observe the performance change
    num_epochs_needed = [np.argmax(sim_list[i] > 0.5) for i in range(len(sim_list))]
    hard2fit = np.argsort(num_epochs_needed) # indices sort training efforts by ascending order
    hard2fit = np.delete(hard2fit, np.where(np.asarray(num_epochs_needed)[hard2fit] == 39)[0])
    hard2fit = np.delete(hard2fit, np.where(np.asarray(num_epochs_needed)[hard2fit] == 0)[0])

    top_hard2fit = np.asarray(hard2fit[-int(len(sim_list)*0.05):])
    top_easy2fit = np.asarray(hard2fit[:int(len(sim_list)*0.05)])

    plt.hist(num_epochs_needed)
    plt.show()
    plt.hist(np.asarray(num_epochs_needed)[top_easy2fit])
    plt.show()
    plt.hist(np.asarray(num_epochs_needed)[top_hard2fit])
    plt.show()

    np.save(os.path.join('hard_samples_ind',
                         'cub_ProxyNCA_pfix_easy_fit.npy'), top_easy2fit)
    np.save(os.path.join('hard_samples_ind',
                         'cub_ProxyNCA_pfix_hard_fit.npy'), top_hard2fit)