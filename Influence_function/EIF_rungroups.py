
import os
from Influence_function.EIF_utils import *
from Influence_function.IF_utils import *
from Influence_function.influence_function import MCScalableIF
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':

    sz_embedding = 512; epoch = 40; test_crop = False
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'inshop'; config_name = 'inshop'; seed = 4

    # loss_type = 'SoftTriple'; dataset_name = 'cub'; config_name = 'cub'; seed = 3
    # loss_type = 'SoftTriple'; dataset_name = 'cars'; config_name = 'cars'; seed = 4
    loss_type = 'SoftTriple'; dataset_name = 'inshop'; config_name = 'inshop'; seed = 3

    IS = MCScalableIF(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)
    #
    confusion_class_pairs = IS.get_confusion_class_pairs()
    for pair_idx in range(len(confusion_class_pairs)):
        if os.path.exists("Influential_data/{}_{}_helpful_testcls{}.npy".format(IS.dataset_name, IS.loss_type, pair_idx)):
            print('skip')
            continue
        '''Step 1: Get deltaD_deltaL'''
        mean_deltaL_deltaD = IS.MC_estimate_group(confusion_class_pairs[pair_idx], num_thetas=1, steps=50)

        '''Step 2: Calc influence functions'''
        influence_values = np.asarray(mean_deltaL_deltaD)
        training_sample_by_influence = influence_values.argsort()  # ascending
        # IS.viz_samples(IS.dl_tr, training_sample_by_influence[:10])  # helpful
        # IS.viz_samples(IS.dl_tr, training_sample_by_influence[-10:])  # harmful

        helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful

        np.save("Influential_data/{}_{}_helpful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx),
                helpful_indices)
        np.save("Influential_data/{}_{}_harmful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx),
                harmful_indices)

    exit()

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
                 '{}_confusion_{}'.format(loss_type, wrong_cls),
                 2, 0,
                 1, dataset_name,
                 dataset_name, 512, seed)))
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After inter-class distance: ", inter_dist_after)

    exit()

