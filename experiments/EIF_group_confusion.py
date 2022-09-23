
import os
from Influence_function.EIF_utils import *
from Influence_function.IF_utils import *
from Influence_function.influence_function import EIF
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running EIF on groups of confusion pairs')
    parser.add_argument('--dataset', default='cub')
    parser.add_argument('--loss-type', default='SoftTriple', type=str)
    parser.add_argument('--seed', default=3, type=int)

    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cub';  config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cars'; config_name = 'cars_ProxyNCA_prob_orig'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'inshop'; config_name = 'inshop_ProxyNCA_prob_orig'; seed = 4

    # loss_type = 'SoftTriple'; dataset_name = 'cub'; config_name = 'cub_SoftTriple'; seed = 3
    # loss_type = 'SoftTriple'; dataset_name = 'cars'; config_name = 'cars_SoftTriple'; seed = 4
    # loss_type = 'SoftTriple'; dataset_name = 'inshop'; config_name = 'inshop_SoftTriple'; seed = 3
    args = parser.parse_args()

    sz_embedding = 512; epoch = 40; test_crop = False; topk_cls = 30
    data_transform_config = 'dataset/config.json'; model_arch = 'ResNet'
    config_name = args.dataset + '_' + args.loss_type

    IS = EIF(dataset_name=args.dataset, seed=args.seed, loss_type=args.loss_type,
             config_name=config_name, data_transform_config=data_transform_config,
             test_crop=test_crop, sz_embedding=sz_embedding, epoch=epoch,
             model_arch=model_arch, mislabel_percentage=0.1)

    '''Find influential training samples'''
    confusion_class_pairs = IS.get_confusion_class_pairs(topk_cls=topk_cls)
    for pair_idx in range(len(confusion_class_pairs)):
        if os.path.exists("Influential_data/{}_{}_helpful_testcls{}.npy".format(IS.dataset_name, IS.loss_type, pair_idx, topk_cls)):
            print('skip')
            continue
        '''Step 1: Get deltaD_deltaL'''
        mean_deltaL_deltaD = IS.EIF_for_groups_confusion(confusion_class_pairs[pair_idx], num_thetas=1, steps=50)

        '''Step 2: Calc influence functions'''
        influence_values = np.asarray(mean_deltaL_deltaD)
        training_sample_by_influence = influence_values.argsort()  # ascending

        helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful

        np.save("Influential_data/{}_{}_helpful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx, topk_cls),
                helpful_indices)
        np.save("Influential_data/{}_{}_harmful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx, topk_cls),
                harmful_indices)
    #

    '''Actually train with downweighted harmful and upweighted helpful training'''
    for pair_idx, class_pair in enumerate(confusion_class_pairs):
        wrong_cls = class_pair[0][0]
        weight_path = 'models/dvi_data_{}_{}_loss{}_2_0/ResNet_512_Model/Epoch_1/{}_{}_trainval_{}_{}.pth'.format(
                        args.dataset, args.seed,
                       '{}_confusion_{}'.format(args.loss_type, wrong_cls),
                        args.dataset, args.dataset, 512, args.seed)

        if os.path.exists(weight_path):
            print("skip")
            continue

        os.system("python train_sample_reweight.py --dataset {} \
                --loss-type {}_confusion_{} \
                --helpful Influential_data/{}_{}_helpful_testcls{}.npy \
                --harmful Influential_data/{}_{}_harmful_testcls{}.npy \
                --model_dir {} \
                --helpful_weight 2 --harmful_weight 0 \
                --seed {} --config config/{}_reweight_{}.json".format(IS.dataset_name,
                                                                   IS.loss_type, wrong_cls,
                                                                   IS.dataset_name, IS.loss_type, pair_idx,
                                                                   IS.dataset_name, IS.loss_type, pair_idx,
                                                                   IS.model_dir,
                                                                   IS.seed, IS.dataset_name, IS.loss_type))

    '''Get confusion distance (before VS after)'''
    IS.model = IS._load_model()  # reload the original weights
    features = IS.get_test_features()

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
                       args.dataset, args.seed,
                      '{}_confusion_{}'.format(args.loss_type, wrong_cls),
                       args.dataset, args.dataset, 512, args.seed)

        IS.model.load_state_dict(torch.load(weight_path))
        inter_dist_after, _ = grad_confusion(IS.model, features, wrong_cls, confuse_classes,
                                             IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
        print("After d(G_p): ", inter_dist_after)


