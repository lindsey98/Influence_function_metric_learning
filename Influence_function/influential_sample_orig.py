from Influence_function.influential_sample import InfluentialSample
from Influence_function.influence_function import grad_confusion
import os
from Influence_function.influence_function_orig import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    # dataset_name = 'cub';  config_name = 'cub'; seed = 0
    dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)

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
                 'ProxyNCA_prob_orig_confusion_{}_baseline'.format(wrong_cls),
                 2, 0,
                 1, dataset_name, dataset_name, 512, seed)))
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After inter-class distance: ", inter_dist_after)

        # reload weights as new
        IS.model.load_state_dict(torch.load(
                  'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(dataset_name, seed,
                   'ProxyNCA_prob_orig_confusion_{}_baseline'.format(wrong_cls),
                   0, 2,
                   1,
                   dataset_name, dataset_name, 512, seed)))
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After inter-class distance reverse: ", inter_dist_after)
    exit()

    '''Step 1: Get grad(test)'''
    train_features = IS.get_train_features()
    test_features = IS.get_features()  # (N, 2048)
    confusion_class_pairs = IS.get_confusion_class_pairs()
    for pair_idx, pair in enumerate(confusion_class_pairs):
        wrong_cls = pair[0][0]
        confused_classes = [x[1] for x in pair]
        inter_dist, v = grad_confusion(IS.model, test_features, wrong_cls, confused_classes,
                                       IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)  # dD/dtheta
        torch.save(v, os.path.join('Influential_data_baselines', 'grad_test_{}_{}_{}.pth'.format(IS.dataset_name, IS.loss_type, wrong_cls)))

        '''Step 2: Get H^-1 grad(test)'''
        ihvp = inverse_hessian_product(IS.model, IS.criterion, v, IS.dl_tr, scale=500, damping=0.01)

        '''Step 3: Get influential indices, i.e. grad(test) H^-1 grad(train), save'''
        influence_values = calc_influential_func(IS=IS, train_features=train_features, inverse_hvp_prod=ihvp)
        influence_values = np.asarray(influence_values).flatten()
        training_sample_by_influence = influence_values.argsort()  # ascending
        # IS.viz_sample(IS.dl_tr, training_sample_by_influence[:10])  # harmful
        # IS.viz_sample(IS.dl_tr, training_sample_by_influence[-10:])  # helpful

        helpful_indices = np.where(influence_values > 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values < 0)[0]  # cache all harmful
        np.save("Influential_data_baselines/{}_{}_helpful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx),
                helpful_indices)
        np.save("Influential_data_baselines/{}_{}_harmful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx),
                harmful_indices)