from Influence_function.influential_sample import InfluentialSample
from Influence_function.influence_function import grad_confusion
import os
from Influence_function.influence_function_orig import *

if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, test_crop, sz_embedding, epoch)

    '''Step 1: Get grad(train)'''
    # train_features = IS.get_train_features()
    # grad4train = grad_loss(IS.model, IS.criterion, train_features, IS.train_label)
    # with open('Influential_data_baselines/{}_{}_grad4train.pkl'.format(IS.dataset_name, IS.loss_type), 'wb') as handle:
    #     pickle.dump(grad4train, handle)

    with open('Influential_data_baselines/{}_{}_grad4train.pkl'.format(IS.dataset_name, IS.loss_type), 'rb') as handle:
        grad4train = pickle.load(handle)

    '''Step 2: Get grad(test)'''
    test_features = IS.get_features()  # (N, 2048)
    confusion_class_pairs = IS.get_confusion_class_pairs()
    for pair_idx, pair in enumerate(confusion_class_pairs):
        wrong_cls = pair[0][0]
        confused_classes = [x[1] for x in pair]
        inter_dist, v = grad_confusion(IS.model, test_features, wrong_cls, confused_classes,
                                       IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)  # dD/dtheta
        torch.save(v, os.path.join('Influential_data_baselines', 'grad_test_{}_{}_{}.pth'.format(IS.dataset_name, IS.loss_type, wrong_cls)))

        '''Step 3: Get H^-1 grad(test)'''
        ihvp = inverse_hessian_product(IS.model, IS.criterion, v, IS.dl_tr, scale=500, damping=0.01)

        '''Step 4: Get influential indices, i.e. grad(test) H^-1 grad(train), save'''
        influence_values = calc_influential_func(inverse_hvp_prod=ihvp, grad_alltrain=grad4train)
        influence_values = np.asarray(influence_values).flatten()
        training_sample_by_influence = influence_values.argsort()  # ascending
        IS.viz_sample(IS.dl_tr, training_sample_by_influence[:10])  # harmful
        IS.viz_sample(IS.dl_tr, training_sample_by_influence[-10:])  # helpful

        helpful_indices = np.where(influence_values > 0)[0]  # cache all helpful
        harmful_indices = np.where(influence_values < 0)[0]  # cache all harmful
        np.save("Influential_data_baselines/{}_{}_helpful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx),
                helpful_indices)
        np.save("Influential_data_baselines/{}_{}_harmful_testcls{}".format(IS.dataset_name, IS.loss_type, pair_idx),
                harmful_indices)
    pass