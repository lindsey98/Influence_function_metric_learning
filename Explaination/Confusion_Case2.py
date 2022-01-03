
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from PIL import Image
from Influence_function.influential_sample import InfluentialSample
from Explaination.CAM_methods import *
from Influence_function.influence_function import *
from Explaination.background_removal import remove_background
import utils
import dataset
from torchvision import transforms
from dataset.utils import RGBAToRGB, ScaleIntensities
from utils import overlay_mask
from utils import predict_batchwise
from evaluation import assign_by_euclidian_at_k_indices
import sklearn
from evaluation.pumap import prepare_data, get_wrong_indices
from Explaination.Confusion_Case1 import DistinguishFeat
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


if __name__ == '__main__':

    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    test_crop = False

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, test_crop)

    '''Analyze confusing features for all confusion classes'''
    '''Step 1: Find pairs that human thinks dissimilar but model thinks similar'''
    # first we find what are the top wrong classes -> do not find samples from these classes because case 2 are perhaps mostly outliers
    _, top20_wrong_classes, wrong_labels, wrong_preds = get_wrong_indices(DF.testing_embedding,
                                                                          DF.testing_label,
                                                                          topk=20)
    # visualize the sample itself (it is wrong), its neighbor
    pred = DF.testing_nn_label.flatten()
    label = DF.testing_label.flatten()
    nn_indices = DF.testing_nn_indices.flatten()

    wrong_indices = ((pred != label) * (1 - torch.isin(label, torch.from_numpy(top20_wrong_classes)).to(torch.float))).nonzero()
    confuse_indices = nn_indices[wrong_indices]
    assert len(confuse_indices) == len(wrong_indices)

    # TODO its neighbor within the same class
    distances = sklearn.metrics.pairwise.pairwise_distances(DF.testing_embedding) # (N_test, N_test)
    same_cls_mask = (DF.testing_label[:, None] == DF.testing_label).detach().cpu().numpy().nonzero()
    distances[same_cls_mask[0], same_cls_mask[1]] = distances.max() + 1
    nn_indices_same_cls = np.argsort(distances, axis = 1)[:, 1]
    wrong_nn_indices_same_cls = nn_indices_same_cls[wrong_indices]
    assert len(confuse_indices) == len(wrong_nn_indices_same_cls)