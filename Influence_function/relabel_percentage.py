import os
import matplotlib.pyplot as plt
import numpy as np
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
    # dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
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

    ct_relabel = 0
    ct_relabel_sig = 0
    centrailized_relabel = []
    diversified_relabel = []
    individual_relabel = []
    for cls in range(IS.dl_tr.dataset.nb_classes()):
        if np.sum(relabel_dist[cls] != cls) > 0:
            ct_relabel += 1
            percentage = np.sum(relabel_dist[cls] != cls) / len(relabel_dist[cls])
            if percentage >= 0.1:
                # print('Cls {} Percentage {}'.format(cls, percentage))
                ct_relabel_sig += 1
                relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) \
                                              for c in range(IS.dl_tr.dataset.nb_classes())])
                high_freq_relabel_cls = relabel_freq.argsort()[::-1]
                if relabel_freq[high_freq_relabel_cls[1]] >= 0.9 * percentage: # centralized
                    centrailized_relabel.append(cls)
                else:
                    diversified_relabel.append(cls)
            elif percentage <= 0.05 and percentage > 0:
                individual_relabel.append(cls)

    print('Total number of classes {}'.format(IS.dl_tr.dataset.nb_classes()))
    print('Total number of classes with relabelling suggestion {}'.format(ct_relabel))
    print('Total number of classes with significant (>=10%) relabelling suggestion {}'.format(ct_relabel_sig))
    print('Centralized', centrailized_relabel)
    print('Diversified', diversified_relabel)
    print('Individual', individual_relabel)

    central_entropy = []
    for cls in centrailized_relabel:
        relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) for c in range(IS.dl_tr.dataset.nb_classes())])
        high_freq_relabel_cls = relabel_freq.argsort()[::-1]
        relabel_freq_ = relabel_freq[relabel_freq.argsort()[::-1][1:][relabel_freq[relabel_freq.argsort()[::-1][1:]] > 0]]
        relabel_entropy = -sum(relabel_freq_ * np.log(relabel_freq_))
        central_entropy.append(relabel_entropy)
    print(central_entropy)

    entropy = []
    for cls in diversified_relabel:
        relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) for c in range(IS.dl_tr.dataset.nb_classes())])
        high_freq_relabel_cls = relabel_freq.argsort()[::-1]
        relabel_freq_ = relabel_freq[relabel_freq.argsort()[::-1][1:][relabel_freq[relabel_freq.argsort()[::-1][1:]] > 0]]
        relabel_entropy = -sum(relabel_freq_ * np.log(relabel_freq_))
        entropy.append(relabel_entropy)
    print(entropy)

    # CUB
    # Centrailized [29, 53]
    # Diversified [58, 61, 96]
    # Individual [1, 2, 8]

    # CARS
    # Cntrailized [12, 21, 45]
    # Diversified [1, 13, 15]
    # Individual [2, 4, 6]

    # InShop
    # Centrolized [23, 24, 25]
    # Diversified [103, 194, 247]
    # Individual [48, 611, 682]

    # cls = 682
    # cls_indices = np.where(np.asarray(IS.dl_tr.dataset.ys) == cls)[0]
    # relabel_as_classes = relabel_dict[cls_indices]

    # relabel_freq = np.asarray([np.sum(relabel_dist[cls] == c) / len(relabel_dist[cls]) for c in range(IS.dl_tr.dataset.nb_classes())])
    # high_freq_relabel_cls = relabel_freq.argsort()[::-1]

    # for topk in range(1, IS.dl_tr.dataset.nb_classes()):
    #     if relabel_freq[high_freq_relabel_cls[topk]] == 0:
    #         break
    #     # top-1 class
    #     topk_relabel_class = np.arange(IS.dl_tr.dataset.nb_classes())[high_freq_relabel_cls[topk]]
    #     interest_indices = np.where(relabel_as_classes == topk_relabel_class)[0]
    #     vis_indices = cls_indices[interest_indices]
    #     fig = plt.figure(figsize=(30, 15))
    #     for kk in range(min(len(vis_indices), 10)):
    #         plt.subplot(2, 10, kk+1)
    #         img = read_image(IS.dl_tr.dataset.im_paths[vis_indices[kk]])
    #         plt.imshow(to_pil_image(img))
    #         plt.title("Ind = {} Cls = {}".format(vis_indices[kk], cls))
    #         plt.axis('off')
    #         # plt.tight_layout()
    #
    #     ind_cls = np.where(np.asarray(IS.dl_tr.dataset.ys) == topk_relabel_class)[0]
    #     for i in range(min(5, len(ind_cls))):
    #         plt.subplot(2, 10, i + 11)
    #         img = read_image(IS.dl_tr.dataset.im_paths[ind_cls[i]])
    #         plt.imshow(to_pil_image(img))
    #         plt.title('Ind = {} Cls = {}'.format(ind_cls[i], topk_relabel_class))
    #         plt.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()

    # for cls in range(IS.dl_tr.dataset.nb_classes()):
    #     if len(np.unique(relabel_dist[cls])) == 5:
    #         fig = plt.figure(figsize=(30, 15))
    #         for kk, unique_cls in enumerate(np.unique(relabel_dist[cls])):
    #             plt.subplot(1, 5, kk+1)
    #             ind = np.where(np.asarray(IS.dl_tr.dataset.ys) == unique_cls)[0][0]
    #             img = read_image(IS.dl_tr.dataset.im_paths[ind])
    #             plt.imshow(to_pil_image(img))
    #             # plt.title("Ind = {}".format(ind))
    #             plt.axis('off')
    #             plt.tight_layout()
    #         plt.savefig("Inshop_{}_{}_{}_{}_{}.png".format(np.unique(relabel_dist[cls])[0],
    #                                                        np.unique(relabel_dist[cls])[1],
    #                                                        np.unique(relabel_dist[cls])[2],
    #                                                        np.unique(relabel_dist[cls])[3],
    #                                                        np.unique(relabel_dist[cls])[4]),bbox_inches='tight')
    #
    #


