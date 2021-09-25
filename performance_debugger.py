import utils
import time
from utils import predict_batchwise
import evaluation
import torch
import numpy as np

def performance(model, dl_ev, k = 1):

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dl_ev)
    print('done collecting prediction')

    # get predictions by assigning nearest k neighbors with euclidian
    indices, Y = evaluation.assign_by_euclidian_at_k_indices(X, T, k) # neighbor indices and their corresponding class

    correct = [(t in y[:k]) for t, y in zip(T, Y)]

    return indices, Y, np.asarray(correct)

def performance_diff(dl_ev,
                     correct1, correct2,
                     indices1, indices2,
                     Y1, Y2,
                     direction=1):
    if direction == 1:
        diff_indices = np.where((correct1==True)&(correct2==False))[0]
    elif direction == 2:
        diff_indices = np.where((correct2==True)&(correct1==False))[0]
    else:
        raise NotImplementedError

    samples = [dl_ev.dataset.__getitem__(ind)[0].permute(1, 2, 0).numpy() for ind in diff_indices]
    label4samples = [dl_ev.dataset.__getitem__(ind)[1] for ind in diff_indices]

    nn1 = indices1[diff_indices, :]
    nn2 = indices2[diff_indices, :]
    nn1_samples = []
    label4nn1_samples = []
    for x in range(len(nn1)):
        nn1_sample_this = []
        nn1_sample_label_this = []
        for ind in range(nn1.shape[1]):
            nn1_sample_this.append(dl_ev.dataset.__getitem__(nn1[x, ind])[0].permute(1, 2, 0).numpy())
            nn1_sample_label_this.append(dl_ev.dataset.__getitem__(nn1[x, ind])[1])
        nn1_samples.append(nn1_sample_this)
        label4nn1_samples.append(nn1_sample_label_this)

    nn2_samples = []
    label4nn2_samples = []
    for x in range(len(nn2)):
        nn2_sample_this = []
        nn2_sample_label_this = []
        for ind in range(nn2.shape[1]):
            nn2_sample_this.append(dl_ev.dataset.__getitem__(nn2[x, ind])[0].permute(1, 2, 0).numpy())
            nn2_sample_label_this.append(dl_ev.dataset.__getitem__(nn2[x, ind])[1])
        nn2_samples.append(nn2_sample_this)
        label4nn2_samples.append(nn2_sample_label_this)

    return samples, label4samples, nn1_samples, label4nn1_samples, nn2_samples, label4nn2_samples


def visualize(samples, label4samples, nn1_samples, label4nn1_samples, nn2_samples, label4nn2_samples):
    # (image -- 1st neighbor from model 1 -- 1st neighbor from model 2) x 3


if __name__ == '__main__':
    print([(y % 10 == 0) for y in range(100)])

