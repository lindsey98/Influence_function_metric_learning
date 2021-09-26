import math

import utils
import time
from utils import predict_batchwise
import evaluation
import torch
import numpy as np
import matplotlib.pyplot as plt
from parametric_umap import prepare_data
from networks import Feat_resnet50_max_n
from torch import nn

def performance(model, dl_ev, k = 1):

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dl_ev)
    print('done collecting prediction')

    # get predictions by assigning nearest k neighbors with euclidian
    indices, Y = evaluation.assign_by_euclidian_at_k_indices(X, T, k) # neighbor indices and their corresponding class
    Y = torch.from_numpy(Y)
    correct = [(t in y[:k]) for t, y in zip(T, Y)]

    return indices, Y, np.asarray(correct)

def performance_diff(dl_ev,
                     correct1, correct2,
                     indices1, indices2,
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


def visualize(samples, label4samples,
              nn1_samples, label4nn1_samples,
              nn2_samples, label4nn2_samples):
    # (image -- 1st neighbor from model 1 -- 1st neighbor from model 2) x 3
    for i in range(math.ceil(len(samples)/3)):
        fig, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(samples[i*3])
        axs[0, 0].set_title('Label {}'.format(label4samples[i*3]))
        axs[0, 1].imshow(nn1_samples[i*3])
        axs[0, 1].set_title('Label {}'.format(label4nn1_samples[i*3]))
        axs[0, 2].imshow(nn2_samples[i*3])
        axs[0, 2].set_title('Label {}'.format(label4nn2_samples[i*3]))

        axs[1, 0].imshow(samples[i*3+1])
        axs[1, 0].set_title('Label {}'.format(label4samples[i*3+1]))
        axs[1, 1].imshow(nn1_samples[i*3+1])
        axs[1, 1].set_title('Label {}'.format(label4nn1_samples[i*3+1]))
        axs[1, 2].imshow(nn2_samples[i*3+1])
        axs[1, 2].set_title('Label {}'.format(label4nn2_samples[i*3+1]))


        axs[2, 0].imshow(samples[i*3+2])
        axs[2, 0].set_title('Label {}'.format(label4samples[i*3+2]))
        axs[2, 1].imshow(nn1_samples[i*3+2])
        axs[2, 1].set_title('Label {}'.format(label4nn1_samples[i*3+2]))
        axs[2, 2].imshow(nn2_samples[i*3+2])
        axs[2, 2].set_title('Label {}'.format(label4nn2_samples[i*3+2]))

        plt.show()
        break


if __name__ == '__main__':
    dataset_name = 'logo2k'
    sz_embedding = 2048
    folder1 = 'dvi_data_logo2k_False_t0.1_proxy1_tau0.0'
    folder2 = 'dvi_data_logo2k_False_t0.1_proxy2_tau0.2'
    dl_tr, dl_ev = prepare_data(data_name=dataset_name, root=folder1, save=False)

    # load model
    feat = Feat_resnet50_max_n()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, sz_embedding)

    model1 = torch.nn.Sequential(feat, emb)
    model1 = nn.DataParallel(model1)
    model1.cuda()
    model_dir1 = '{}/ResNet_{}_Model'.format(folder1, sz_embedding)
    model1.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_{}_0.pth'.format(model_dir1, 40, dataset_name, dataset_name, sz_embedding)))
    indices1, Y1, correct1 = performance(model1, dl_tr, k = 1)

    model2 = torch.nn.Sequential(feat, emb)
    model2 = nn.DataParallel(model2)
    model2.cuda()
    model_dir2 = '{}/ResNet_{}_Model'.format(folder2, sz_embedding)
    model2.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_{}_0.pth'.format(model_dir2, 40, dataset_name, dataset_name, sz_embedding)))
    indices2, Y2, correct2 = performance(model2, dl_tr, k = 1)

    samples, label4samples, nn1_samples, label4nn1_samples, nn2_samples, label4nn2_samples = performance_diff(dl_tr,
                                                                                                             correct1, correct2,
                                                                                                             indices1, indices2,
                                                                                                             direction=1)
    visualize(samples, label4samples,
              nn1_samples, label4nn1_samples,
              nn2_samples, label4nn2_samples)
