
import os
from Influence_function.ScalableIF_utils import *
from Influence_function.IF_utils import *
from Influence_function.influence_function import ScalableIF
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == '__main__':

    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    # dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = ScalableIF(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)

    '''Other: get confusion (before VS after)'''
    # FIXME: inter class distance should be computed based on original confusion pairs
    #  confusion class pairs is computed with original weights, then we do weight reload
    IS.model = IS._load_model()  # reload the original weights
    features = IS.get_features()

    confusion_class_pairs = IS.get_confusion_class_pairs()
    for pair_idx in range(len(confusion_class_pairs)):
        print('Pair index', pair_idx)
        wrong_cls = confusion_class_pairs[pair_idx][0][0]
        confuse_classes = [x[1] for x in confusion_class_pairs[pair_idx]]

        IS.model = IS._load_model() # reload the original weights
        inter_dist_orig, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                            IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("Original inter-class distance: ", inter_dist_orig)

        # reload weights as new
        IS.model.load_state_dict(torch.load(
                'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(dataset_name, seed,
                 'ProxyNCA_prob_orig_confusion_{}'.format(wrong_cls),
                 2, 0,
                 1, dataset_name,
                 dataset_name, 512, seed)))
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After inter-class distance: ", inter_dist_after)

        # reload weights as new
        IS.model.load_state_dict(torch.load(
                  'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(dataset_name,
                   seed,
                   'ProxyNCA_prob_orig_confusion_{}'.format(wrong_cls),
                   0, 2,
                   1,
                   dataset_name,
                   dataset_name,
                   512, seed)))
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After inter-class distance reverse: ", inter_dist_after)
    exit()
    #
    '''Step 1: Cache all confusion gradient to parameters'''
    confusion_class_pairs = IS.get_confusion_class_pairs()
    IS.agg_get_theta(confusion_class_pairs)

    '''Step 2: Cache training class loss changes'''
    for cls_idx in range(len(confusion_class_pairs)):
        i = confusion_class_pairs[cls_idx][0][0]
        theta_dict = torch.load("Influential_data/{}_{}_confusion_theta_test_{}.pth".format(IS.dataset_name, IS.loss_type, i))
        theta = theta_dict['theta']
        theta_hat = theta_dict['theta_hat']
        IS.cache_grad_loss_train_all(theta, theta_hat, cls_idx)

    '''Step 3: Calc influence functions'''
    for cls_idx in range(len(confusion_class_pairs)):
        IS.agg_influence_func(cls_idx)
    exit()

