
import os
from Influence_function.influential_sample import InfluentialSample
from Influence_function.influence_function import *
from evaluation.pumap import get_wrong_indices
from evaluation import assign_by_euclidian_at_k_indices
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

if __name__ == '__main__':

    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    test_crop = False

    IS = InfluentialSample(dataset_name, seed, loss_type, config_name, test_crop)

    '''Analyze confusing features for all confusion classes'''
    '''Step 1: Get all wrong pairs'''
    test_nn_indices, test_nn_label = assign_by_euclidian_at_k_indices(IS.testing_embedding, IS.testing_label, 1)
    wrong_indices = (test_nn_label.flatten() != IS.testing_label.detach().cpu().numpy().flatten()).nonzero()[0]
    confuse_indices = test_nn_indices.flatten()[wrong_indices]
    assert len(wrong_indices) == len(confuse_indices)
    base_dir = 'Confuse_pair_influential_data/{}'.format(IS.dataset_name)
    os.makedirs(base_dir, exist_ok=True)

    '''Step 2: Save influential samples indices'''
    # for kk in tqdm(range(min(len(wrong_indices), 50))):
    #     wrong_ind = wrong_indices[kk]
    #     confuse_ind = confuse_indices[kk]
    #     # sanity check: IS.viz_2sample(IS.dl_ev, wrong_ind, confuse_ind)
    #     training_sample_by_influence, influence_values = IS.single_influence_func([wrong_ind], [confuse_ind])
    #     helpful_indices = np.where(influence_values < 0)[0]
    #     harmful_indices = np.where(influence_values > 0)[0]
    #     np.save('./{}/Allhelpful_indices_{}_{}'.format(base_dir, wrong_ind, confuse_ind), helpful_indices)
    #     np.save('./{}/Allharmful_indices_{}_{}'.format(base_dir, wrong_ind, confuse_ind), harmful_indices)

    '''Step 3: Train the model'''
    # Run in shell
    # for kk in tqdm(range(min(len(wrong_indices), 50))):
    #     wrong_ind = wrong_indices[kk]
    #     confuse_ind = confuse_indices[kk]
    #     #  Normal training
    #     os.system("python train_sample_reweight.py --dataset {} \
    #                     --loss-type ProxyNCA_pfix_confusion_{}_{}_Allsamples \
    #                     --helpful {}/Allhelpful_indices_{}_{}.npy \
    #                     --harmful {}/Allharmful_indices_{}_{}.npy \
    #                     --model_dir {} \
    #                     --helpful_weight 2 --harmful_weight 0 \
    #                     --seed {} --config config/{}_reweight.json".format(IS.dataset_name,
    #                                                                        wrong_ind, confuse_ind,
    #                                                                        base_dir, wrong_ind, confuse_ind,
    #                                                                        base_dir, wrong_ind, confuse_ind,
    #                                                                        IS.model_dir,
    #                                                                        IS.seed,
    #                                                                        IS.dataset_name))
    #     # reverse training
    #     os.system("python train_sample_reweight.py --dataset {} \
    #                     --loss-type ProxyNCA_pfix_confusion_{}_{}_Allsamples \
    #                     --helpful {}/Allhelpful_indices_{}_{}.npy \
    #                     --harmful {}/Allharmful_indices_{}_{}.npy \
    #                     --model_dir {} \
    #                     --helpful_weight 0 --harmful_weight 2 \
    #                     --seed {} --config config/{}_reweight.json".format(IS.dataset_name,
    #                                                                        wrong_ind, confuse_ind,
    #                                                                        base_dir, wrong_ind, confuse_ind,
    #                                                                        base_dir, wrong_ind, confuse_ind,
    #                                                                        IS.model_dir,
    #                                                                        IS.seed,
    #                                                                        IS.dataset_name))
    #     exit()

    '''Step 4: Sanity check: Whether the confusion pairs are pulled far apart, Whether the confusion samples is pulled closer to correct neighbor'''
    for kk in range(min(len(wrong_indices), 50)):
        wrong_ind = wrong_indices[kk]
        confuse_ind = confuse_indices[kk]
        new_weight_path = 'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
                           dataset_name,
                           seed,
                           'ProxyNCA_pfix_confusion_{}_{}_Allsamples'.format(
                           wrong_ind, confuse_ind),
                           2, 0,
                           1, dataset_name,
                           dataset_name,
                           512, seed) # reload weights as new

        new_reverse_weight_path = 'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
                                   dataset_name,
                                   seed,
                                   'ProxyNCA_pfix_confusion_{}_{}_Allsamples'.format(
                                   wrong_ind, confuse_ind),
                                   0, 2,
                                   1, dataset_name,
                                   dataset_name,
                                   512, seed) # reload weights as new

        IS.model = IS._load_model()  # reload the original weights
        new_features = IS.get_features()
        inter_dist_orig, _ = grad_confusion_pair(IS.model, new_features, [wrong_ind], [confuse_ind])
        print("Original distance: ", inter_dist_orig)

        IS.model.load_state_dict(torch.load(new_weight_path))
        inter_dist_after, _ = grad_confusion_pair(IS.model, new_features, [wrong_ind], [confuse_ind])
        print("After distance: ", inter_dist_after)

        IS.model.load_state_dict(torch.load(new_reverse_weight_path))
        inter_dist_after, _ = grad_confusion_pair(IS.model, new_features, [wrong_ind], [confuse_ind])
        print("After distance reverse: ", inter_dist_after)
        exit()
