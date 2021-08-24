import json

import numpy as np


def loss_potential(loss_dict, current_t, rolling_t, ts_diff, ts_loss, ts_ratio, len_training):
    '''
    Compute the hard indices
    :param loss_dict: Dictionary for loss
    :param current_t: current epoch
    :param rolling_t: number of previous rolling epochs
    :param ts_diff: threshold to decide rolling difference is large or not
    :param ts_loss: threshold to determine loss is high or not
    :param ts_ratio: threshold to decide whether hard examples are prevalent
    :param len_training: number of training data
    '''
    # TODO: should get potential for each class
    update = False
    loss_prev = np.stack([loss_dict[str(x)] for x in range(current_t-rolling_t+1, current_t+1)]).T
    assert loss_prev.shape[0] == len_training
    assert loss_prev.shape[1] == min(rolling_t, current_t)

    indices1 = np.where(np.array(loss_dict[str(current_t)]) >= ts_loss)[0] # get high loss indices

    rolling_diff = np.diff(loss_prev)
    indices2 = np.where((np.abs(rolling_diff) <= ts_diff).all(1) == True)[0] # get slow convergence indices

    # get intersection between indices1 and indices2
    indices = set(indices1).intersection(set(indices2))
    ratio = len(indices) / len_training

    # need a new proxy there
    if ratio >= ts_ratio:
        update = True

    return update, indices



if __name__ == '__main__':
    with open('./log/logo2k_logo2k_trainval_2048_0_loss.json', 'rt') as handle:
        loss_dict = json.load(handle)

    update, indices = loss_potential(loss_dict=loss_dict, current_t=5, rolling_t=5, ts_diff=1, ts_loss=0.0,
                   ts_ratio=0.0, len_training=len(loss_dict['0']))