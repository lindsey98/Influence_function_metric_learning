import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from Influence_function.influence_function import MCScalableIF
from explaination.CAM_methods import *
from Influence_function.EIF_utils import *
from Influence_function.IF_utils import *
from utils import overlay_mask
from evaluation import assign_by_euclidian_at_k_indices
import sklearn
import pickle
from utils import evaluate
from Influence_function.sample_relabel import SampleRelabel, kNN_label_pred
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':

    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = SampleRelabel(dataset_name, seed, loss_type, config_name, test_crop)

    relabel_dict = []
    unique_labels, unique_counts = torch.unique(IS.train_label, return_counts=True)
    median_shots_percls = unique_counts.median().item()
    _, prob_relabel = kNN_label_pred(query_indices=np.arange(len(IS.dl_tr.dataset)), embeddings=IS.train_embedding,
                                     labels=IS.train_label,
                                     nb_classes=IS.dl_tr.dataset.nb_classes(), knn_k=median_shots_percls)

    for kk in range(len(IS.dl_tr.dataset)):
        relabel_dict.append(prob_relabel[kk].detach().cpu().numpy().argmax())
    relabel_dict = np.asarray(relabel_dict)

    # Distribution of each class being relabelled as
    relabel_dist = []
    for cls in range(IS.dl_tr.dataset.nb_classes()):
        cls_indices = np.where(np.asarray(IS.dl_tr.dataset.ys) == cls)[0]
        relabel_dist.append(relabel_dict[cls_indices])
    relabel_dist = np.asarray(relabel_dist)

    # for cls in range(IS.dl_tr.dataset.nb_classes()):
    #     if np.sum(relabel_dist[cls] != cls) > 0:
    #         percentage = np.sum(relabel_dist[cls] != cls) / len(relabel_dist[cls])
    #         if percentage >= 0.1:
    #             print('Cls {} Percentage {}'.format(cls, percentage))
    #             plt.bar(np.arange(IS.dl_tr.dataset.nb_classes()),
    #                     height=[np.sum(relabel_dist[cls] == c)/len(relabel_dist[cls]) for c in range(IS.dl_tr.dataset.nb_classes())])
    #             plt.show()

    # relabel_freq = np.asarray([np.sum(relabel_dist[96] == c)/len(relabel_dist[96]) \
    #                            for c in range(IS.dl_tr.dataset.nb_classes())])
    # high_freq_relabel_cls = relabel_freq.argsort()[::-1]
    # print(high_freq_relabel_cls)
    # print(relabel_freq)

    # Cls 29 Percentage 0.1, all relabelled as class 28
    # Cls 53 Percentage 0.1, all relabelled as 13
    # Cls 58 Percentage 0.16666666666666666, and 0.13333333 are relabelled as 65
    # Cls 61 Percentage 0.16666666666666666, 0.11666667 are relabelled as 65
    # Cls 96 Percentage 0.11864406779661017, 0.05084746 are relabelled as 95, 0.05084746 as 94, 0.01694915 as 29
    cls = 29
    cls_indices = np.where(np.asarray(IS.dl_tr.dataset.ys) == cls)[0]






