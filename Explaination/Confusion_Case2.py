
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
from utils import predict_batchwise, predict_batchwise_debug
from evaluation import assign_by_euclidian_at_k_indices, assign_same_cls_neighbor, assign_diff_cls_neighbor
import sklearn
from evaluation.pumap import prepare_data, get_wrong_indices
from Explaination.Confusion_Case1 import DistinguishFeat
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def cache_influential_samples(DF, wrong_index, confuse_index, base_dir):

    os.makedirs(base_dir, exist_ok=True)
    training_sample_by_influence = DF.temporal_influence_func([wrong_index], [confuse_index])
    helpful_indices = training_sample_by_influence[:10]
    harmful_indices = training_sample_by_influence[-10:]
    np.save('./{}/helpful_indices_{}_{}'.format(base_dir, wrong_index, confuse_index), helpful_indices)
    np.save('./{}/harmful_indices_{}_{}'.format(base_dir, wrong_index, confuse_index), harmful_indices)
    exit()

def sanity_check(DF, new_weight_path, wrong_index, confuse_index):
    DF.model = DF._load_model()  # reload the original weights
    new_features = DF.get_features()
    inter_dist_orig, _ = grad_confusion_pair(DF.model, new_features, [wrong_index], [confuse_index])
    print("Original distance: ", inter_dist_orig)

    DF.model.load_state_dict(torch.load(new_weight_path))
    new_features = DF.get_features()
    inter_dist_after, _ = grad_confusion_pair(DF.model, new_features, [wrong_index], [confuse_index])
    print("After distance: ", inter_dist_after)

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
    # _, top20_wrong_classes, _, _ = get_wrong_indices(DF.testing_embedding,
    #                                                  DF.testing_label,
    #                                                  topk=20)

    # visualize the sample itself (it is wrong), its neighbor with distance
    # test_nn_indices_orig, test_nn_label_orig = assign_by_euclidian_at_k_indices(DF.testing_embedding, DF.testing_label, 1)
    # test_nn_indices_orig = torch.from_numpy(test_nn_indices_orig.flatten())
    # test_nn_label_orig = torch.from_numpy(test_nn_label_orig.flatten())

    # wrong_indices = ((test_nn_label_orig != DF.testing_label) * (1 - torch.isin(DF.testing_label, torch.from_numpy(top20_wrong_classes)).to(torch.float))).nonzero()
    # FIXME: I specify
    # wrong_indices = torch.tensor([35, 46, 79, 64, 77, 55, 105, 119, 145, 2102])
    # confuse_indices = test_nn_indices_orig[wrong_indices]
    # assert len(confuse_indices) == len(wrong_indices)

    # also visualize its neighbor within the same class with distance
    # distances = sklearn.metrics.pairwise.pairwise_distances(DF.testing_embedding) # (N_test, N_test)
    # same_cls_mask = (DF.testing_label[:, None] != DF.testing_label).detach().cpu().numpy().nonzero()
    # distances[same_cls_mask[0], same_cls_mask[1]] = distances.max() + 1
    # test_nn_indices_same_cls = np.argsort(distances, axis = 1)[:, 1]
    # test_wrong_nn_indices_same_cls = test_nn_indices_same_cls[wrong_indices]
    # # assert len(confuse_indices) == len(test_wrong_nn_indices_same_cls)
    # print(len(confuse_indices))

    # for wrong_ind, confuse_ind in zip(wrong_indices, confuse_indices):
    #     DF.GradAnalysis(int(DF.testing_label[wrong_ind].item()), int(DF.testing_label[confuse_ind].item()),
    #                      [wrong_ind], [confuse_ind],
    #                      DF.dl_ev, base_dir='Confuse_Vis4Case2')

    '''Step 2: Identify helpful/harmful training samples'''
    wrong_index = 77; confuse_index = 4103
    base_dir = 'Confuse_pair_influential_data'
    new_weight_path = 'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
                           dataset_name,
                           seed,
                           'ProxyNCA_pfix_confusion_{}_{}'.format(
                           wrong_index, confuse_index),
                           10, -10,
                           1, dataset_name,
                           dataset_name,
                           512, seed) # reload weights as new
    #
    DF.viz_2sample(DF.dl_ev, wrong_index, confuse_index)
    # cache_influential_samples(DF, wrong_index, confuse_index, base_dir)

    '''Step 3: Run training in bash'''

    '''Step 4: Sanity check'''
    # sanity_check(DF, new_weight_path, wrong_index, confuse_index)

    '''Step 5: Visualize helpful/harmful'''
    helpful_indices = np.load('./{}/helpful_indices_{}_{}.npy'.format(base_dir, wrong_index, confuse_index))
    harmful_indices = np.load('./{}/harmful_indices_{}_{}.npy'.format(base_dir, wrong_index, confuse_index))
    nn_k = 5 # check 5 neighbors

    # predict training 1st NN (original)
    train_nn_indices_orig, train_nn_label_orig = assign_by_euclidian_at_k_indices(DF.train_embedding, DF.train_label, nn_k)
    # predict training 1st NN within the same class (original)
    train_nn_indices_same_cls_orig = assign_same_cls_neighbor(DF.train_embedding, DF.train_label, nn_k)
    # predict training 1st NN from diff class
    train_nn_indices_diff_cls_orig = assign_diff_cls_neighbor(DF.train_embedding, DF.train_label, nn_k)

    # predict training 1st NN (after training)
    new_model = DF._load_model()
    new_model.load_state_dict(torch.load(new_weight_path))
    train_embedding_curr, train_label, _ = predict_batchwise_debug(new_model, DF.dl_tr)
    train_nn_indices_curr, train_nn_label_curr = assign_by_euclidian_at_k_indices(train_embedding_curr, DF.train_label, nn_k)
    # predict training 1st NN within the same class (after training)
    train_nn_indices_same_cls_curr = assign_same_cls_neighbor(train_embedding_curr, DF.train_label, nn_k)
    # predict training 1st NN from diff class
    train_nn_indices_diff_cls_curr = assign_diff_cls_neighbor(train_embedding_curr, DF.train_label, nn_k)

    # Plot out affected training (its original NN, and its current NN)
    model_orig = DF._load_model()
    model_curr = DF._load_model()
    model_curr.load_state_dict(torch.load(new_weight_path))

    for index in helpful_indices:
        DF.VisTrainNN( wrong_index, confuse_index,
                   index, train_nn_indices_diff_cls_orig[index], train_nn_indices_same_cls_orig[index],
                   train_nn_indices_diff_cls_curr[index], train_nn_indices_same_cls_curr[index],
                   model_orig, model_curr,
                   DF.dl_tr,
                   base_dir)
    #
    # # Plot out helpful training
    # DF.VisTrain(wrong_ind=wrong_index, confusion_ind=confuse_index,
    #             interest_indices=helpful_indices,
    #             orig_NN_indices=train_nn_indices_same_cls_orig.flatten()[helpful_indices],
    #             curr_NN_indices=train_nn_indices_same_cls_curr.flatten()[helpful_indices],
    #             model1=model_orig, model2=model_curr,
    #             dl=DF.dl_tr,
    #             base_dir='Confuse_helpful_train'
    #         )