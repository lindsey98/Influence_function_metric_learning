
import os
from Influence_function.influence_function import OrigIF
from Influence_function.IF_utils import *
from Influence_function.EIF_utils import grad_confusion_pair
from evaluation import assign_by_euclidian_at_k_indices
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 0"

if __name__ == '__main__':

    sz_embedding = 512; epoch = 40; test_crop = False
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'sop'; config_name = 'sop'; seed = 3

    # loss_type = 'SoftTriple'; dataset_name = 'cub'; config_name = 'cub'; seed = 3
    # loss_type = 'SoftTriple'; dataset_name = 'cars'; config_name = 'cars'; seed = 4
    loss_type = 'SoftTriple'; dataset_name = 'inshop'; config_name = 'inshop'; seed = 3

    IS = OrigIF(dataset_name, seed, loss_type, config_name, test_crop)

    '''Analyze confusing features for all confusion classes'''
    '''Step 1: Get all wrong pairs'''
    testing_embedding, testing_label = IS.testing_embedding, IS.testing_label
    test_nn_indices, test_nn_label = assign_by_euclidian_at_k_indices(testing_embedding, testing_label, 1)
    wrong_indices = (test_nn_label.flatten() != testing_label.detach().cpu().numpy().flatten()).nonzero()[0]
    confuse_indices = test_nn_indices.flatten()[wrong_indices]
    print(len(confuse_indices))
    assert len(wrong_indices) == len(confuse_indices)

    base_dir = 'Confuse_pair_influential_data_baselines/{}'.format(IS.dataset_name)
    os.makedirs(base_dir, exist_ok=True)

    '''Step 2: Save influential samples indices for 50 pairs'''
    # train_features = IS.get_train_features()
    # test_features = IS.get_features()  # (N, 2048)
    # for kk in range(min(len(wrong_indices), 50)):
    #     wrong_ind = wrong_indices[kk]
    #     confuse_ind = confuse_indices[kk]
    #     if os.path.exists('./{}/{}_helpful_indices_{}_{}.npy'.format(base_dir, loss_type, wrong_ind, confuse_ind)):
    #         print('skip')
    #         continue
    #     # sanity check: IS.viz_2sample(IS.dl_ev, wrong_ind, confuse_ind)
    #     torch.cuda.empty_cache()
    #     influence_values = IS.single_influence_func_orig(train_features=train_features, test_features=test_features,
    #                                                      wrong_indices=[wrong_ind], confuse_indices=[confuse_ind])
    #
    #     helpful_indices = np.where(influence_values > 0)[0]  # cache all helpful
    #     harmful_indices = np.where(influence_values < 0)[0]  # cache all harmful
    #     np.save('./{}/{}_helpful_indices_{}_{}'.format(base_dir, loss_type, wrong_ind, confuse_ind), helpful_indices)
    #     np.save('./{}/{}_harmful_indices_{}_{}'.format(base_dir, loss_type, wrong_ind, confuse_ind), harmful_indices)
    # exit()

    '''Step 3: Train the model for every pair'''
    # Run in shell
    for kk in tqdm(range(min(len(wrong_indices), 50))):
        wrong_ind = wrong_indices[kk]
        confuse_ind = confuse_indices[kk]
        torch.cuda.empty_cache()
        #  Normal training
        os.system("python train_sample_reweight.py --dataset {} \
                        --loss-type {}_confusion_{}_{}_Allsamples_baseline \
                        --helpful {}/{}_helpful_indices_{}_{}.npy \
                        --harmful {}/{}_harmful_indices_{}_{}.npy \
                        --model_dir {} \
                        --helpful_weight 2 --harmful_weight 0 \
                        --seed {} --config config/{}.json".format(IS.dataset_name,
                                                                   loss_type, wrong_ind, confuse_ind,
                                                                   base_dir, loss_type, wrong_ind, confuse_ind,
                                                                   base_dir, loss_type, wrong_ind, confuse_ind,
                                                                   IS.model_dir,
                                                                   IS.seed,
                                                                   '{}_softtriple_reweight'.format(dataset_name)))


    '''Step 4: Sanity check: Whether the confusion pairs are pulled far apart, Whether the confusion samples is pulled closer to correct neighbor'''
    result_log_file = 'Confuse_pair_influential_data/{}_{}_pairs_baseline.txt'.format(IS.dataset_name, loss_type)
    IS.model = IS._load_model()  # reload the original weights
    new_features = IS.get_features()
    for kk in range(min(len(wrong_indices), 50)):
        wrong_ind = wrong_indices[kk]
        confuse_ind = confuse_indices[kk]
        # Skip written models
        if os.path.exists(result_log_file):
            lines = open(result_log_file).readlines()
            have_written = False
            for l in lines:
                if '{}\t{}'.format(wrong_ind, confuse_ind) in l:
                    have_written = True
                    break
            if have_written:
                continue

        new_weight_path = 'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
                           dataset_name,
                           seed,
                           '{}_confusion_{}_{}_Allsamples_baseline'.format(loss_type, wrong_ind, confuse_ind),
                           2, 0,
                           1, dataset_name,
                           dataset_name,
                           512, seed) # reload weights as new

        IS.model = IS._load_model()  # reload the original weights
        inter_dist_orig, _ = grad_confusion_pair(IS.model, new_features, [wrong_ind], [confuse_ind])

        IS.model.load_state_dict(torch.load(new_weight_path))
        inter_dist_after, _ = grad_confusion_pair(IS.model, new_features, [wrong_ind], [confuse_ind])

        # log results
        with open(result_log_file, 'a+') as f:
            f.write('{}\t{}\t{:.4f}\t{:.4f}\n'.format(wrong_ind, confuse_ind, inter_dist_orig, inter_dist_after))
