
import os
from Influence_function.EIF_utils import *
from Influence_function.IF_utils import *
from Influence_function.influence_function import EIF
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == '__main__':

    sz_embedding = 512; epoch = 40; test_crop = False; topk_cls = 30; data_transform_config = 'dataset/config_bninception.json'; model_arch = 'BnInception'
    loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cub';  config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cars'; config_name = 'cars_ProxyNCA_prob_orig'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'inshop'; config_name = 'inshop_ProxyNCA_prob_orig'; seed = 4

    # loss_type = 'SoftTriple'; dataset_name = 'cub'; config_name = 'cub_SoftTriple'; seed = 3
    # loss_type = 'SoftTriple'; dataset_name = 'cars'; config_name = 'cars_SoftTriple'; seed = 4
    # loss_type = 'SoftTriple'; dataset_name = 'inshop'; config_name = 'inshop_SoftTriple'; seed = 3

    IS = EIF(dataset_name, seed, loss_type, config_name, data_transform_config, test_crop, sz_embedding, epoch, model_arch)
    base_dir = 'Influential_data_BnInception'
    os.makedirs(base_dir, exist_ok=True)

    '''Find influential training samples'''
    confusion_class_pairs = IS.get_confusion_class_pairs(topk_cls=topk_cls)
    # for pair_idx in range(len(confusion_class_pairs)):
    #     if os.path.exists("{}/{}_{}_helpful_testcls{}.npy".format(base_dir, IS.dataset_name, IS.loss_type, pair_idx, topk_cls)):
    #         print('skip')
    #         continue
    #     '''Step 1: Get deltaD_deltaL'''
    #     mean_deltaL_deltaD = IS.MC_estimate_forclasses(confusion_class_pairs[pair_idx], num_thetas=1, steps=50)
    #
    #     '''Step 2: Calc influence functions'''
    #     influence_values = np.asarray(mean_deltaL_deltaD)
    #     training_sample_by_influence = influence_values.argsort()  # ascending
    #
    #     helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
    #     harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful
    #
    #     np.save("{}/{}_{}_helpful_testcls{}".format(base_dir, IS.dataset_name, IS.loss_type, pair_idx, topk_cls),
    #             helpful_indices)
    #     np.save("{}/{}_{}_harmful_testcls{}".format(base_dir, IS.dataset_name, IS.loss_type, pair_idx, topk_cls),
    #             harmful_indices)
    #
    # exit()

    '''Actually train with downweighted harmful and upweighted helpful training'''
    for pair_idx, class_pair in enumerate(confusion_class_pairs):
        wrong_cls = class_pair[0][0]
        weight_path = 'models/dvi_data_{}_{}_loss{}_2_0/ResNet_512_Model/Epoch_1/{}_{}_trainval_{}_{}.pth'.format(
                        dataset_name, seed,
                       '{}_confusion_{}_BnInception'.format(loss_type, wrong_cls),
                        dataset_name, dataset_name, 512, seed)

        if os.path.exists(weight_path):
            print("skip")
            continue

        os.system("python train_sample_reweight_bninception.py --dataset {} \
                --loss-type {}_confusion_{}_BnInception \
                --helpful {}/{}_{}_helpful_testcls{}.npy \
                --harmful {}/{}_{}_harmful_testcls{}.npy \
                --model_dir {} \
                --helpful_weight 2 --harmful_weight 0 \
                --seed {} --config config/{}_reweight_{}_bninception.json".format(IS.dataset_name,
                                                                   IS.loss_type, wrong_cls,
                                                                   base_dir, IS.dataset_name, IS.loss_type, pair_idx,
                                                                   base_dir, IS.dataset_name, IS.loss_type, pair_idx,
                                                                   IS.model_dir,
                                                                   IS.seed, IS.dataset_name, IS.loss_type))

    '''Other: get confusion (before VS after)'''
    # FIXME: inter class distance should be computed based on original confusion pairs
    #  confusion class pairs is computed with original weights, then we do weight reload
    IS.model = IS._load_model()  # reload the original weights
    features = IS.get_test_features()

    confusion_class_pairs = IS.get_confusion_class_pairs(topk_cls=topk_cls)
    for pair_idx in range(len(confusion_class_pairs)):
        print('Pair index', pair_idx)
        wrong_cls = confusion_class_pairs[pair_idx][0][0]
        confuse_classes = [x[1] for x in confusion_class_pairs[pair_idx]]

        IS.model = IS._load_model() # reload the original weights
        inter_dist_orig, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                            IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("Original d(G_p): ", inter_dist_orig)

        # reload weights as new
        weight_path = 'models/dvi_data_{}_{}_loss{}_2_0/ResNet_512_Model/Epoch_1/{}_{}_trainval_{}_{}.pth'.format(
                       dataset_name, seed,
                      '{}_confusion_{}_BnInception'.format(loss_type, wrong_cls),
                       dataset_name, dataset_name, 512, seed)

        IS.model.load_state_dict(torch.load(weight_path))
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After d(G_p): ", inter_dist_after)

