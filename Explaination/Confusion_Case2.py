
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
