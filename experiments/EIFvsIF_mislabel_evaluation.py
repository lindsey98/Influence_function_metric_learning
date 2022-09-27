import os
import torch
from Influence_function.EIF_utils import grad_confusion, calc_loss_train, avg_confusion
from Influence_function.IF_utils import inverse_hessian_product, calc_influential_func_orig, grad_loss
from Influence_function.influence_function import OrigIF, EIF, kNN_label_pred
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from evaluation.recall import assign_by_euclidian_at_k_indices, assign_by_euclidian_at_k
matplotlib.use('TkAgg')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    noisy_level = 0.1
    sz_embedding = 512; epoch = 40; test_crop = False
    loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'cub_noisy';  config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'cars_noisy'; config_name = 'cars_ProxyNCA_prob_orig'; seed = 3
    # loss_type = 'ProxyNCA_prob_orig_noisy_{}'.format(noisy_level); dataset_name = 'inshop_noisy'; config_name = 'inshop_ProxyNCA_prob_orig'; seed = 4

    # loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'cub_noisy';  config_name = 'cub_SoftTriple'; seed = 3
    # loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'cars_noisy';  config_name = 'cars_SoftTriple'; seed = 4
    # loss_type = 'SoftTriple_noisy_{}'.format(noisy_level); dataset_name = 'inshop_noisy';  config_name = 'inshop_SoftTriple'; seed = 3

    '''============================================= Our Empirical Influence function =============================================================='''
    IS = EIF(dataset_name, seed, loss_type, config_name, 'dataset/config.json', test_crop, sz_embedding, epoch, 'ResNet', noisy_level)
    basedir = 'MislabelExp_Influential_data'
    os.makedirs(basedir, exist_ok=True)
    gt_mislabelled_indices = IS.dl_tr.dataset.noisy_indices

    for num_thetas in [1]:
        '''Mislabelled data detection'''
        if os.path.exists("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level)):
            helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
            harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
            influence_values = np.load("{}/{}_{}_influence_values_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level))
        else:
            confusion_class_pairs = IS.get_confusion_class_pairs()

            '''Step 1: Get deltaD_deltaL with the confusion pairs for the top1 frequently wrong testing class'''
            mean_deltaL_deltaD = IS.EIF_for_groups_confusion(confusion_class_pairs[0],
                                                             num_thetas=num_thetas,
                                                             steps=50)
            '''Step 2: Calc influence functions'''
            influence_values = np.asarray(mean_deltaL_deltaD)
            helpful_indices = np.where(influence_values < 0)[0]  # cache all helpful
            harmful_indices = np.where(influence_values > 0)[0]  # cache all harmful

            np.save("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}_step50".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), helpful_indices)
            np.save("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}_step50".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), harmful_indices)
            np.save("{}/{}_{}_influence_values_testcls{}_SIF_theta{}_{}_step50".format(basedir, IS.dataset_name, IS.loss_type, 0, num_thetas, noisy_level), influence_values)

        # mislabelled indices ground-truth
        training_sample_by_influence = influence_values.argsort()[::-1]  # harmful first, followed by helpful
        sorted_influence_values = [influence_values[ct] for ct in training_sample_by_influence]
        training_sample_rank = np.empty_like(training_sample_by_influence)
        training_sample_rank[training_sample_by_influence] = np.arange(len(influence_values))

        overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
        cum_overlap = np.cumsum(overlap)
        ticks_pos = np.arange(0, 1.2, 0.2)
        fraction_data_scanned = [int(x) for x in ticks_pos * len(training_sample_by_influence)]
        fraction_mislabelled_detected = [int(x) for x in ticks_pos * len(gt_mislabelled_indices)]
        # plt.plot(cum_overlap, label='EIF with N_theta = {}'.format(num_thetas))
        # plt.scatter(np.asarray([influence_values[ct] for ct in training_sample_by_influence]),
        #             cum_overlap / len(gt_mislabelled_indices),
        #             label='EIF with N_theta = {}'.format(num_thetas))
        # plt.xlabel("Fraction of training data checked", fontdict={'fontsize': 15})
        # plt.ylabel("Influence values", fontdict={'fontsize': 15})
        # plt.ylabel("Fraction of mislabelled data detected", fontdict={'fontsize': 15})
        # plt.xticks(fraction_data_scanned, [round(x, 1) for x in ticks_pos])
        # plt.yticks(fraction_mislabelled_detected, [round(x, 1) for x in ticks_pos])
        # for tick in fraction_data_scanned:
        #     plt.annotate('EIF={:.4f}'.format(sorted_influence_values[min(tick, len(training_sample_by_influence)-1)]),
        #                  xy=(tick, -1))
    '''============================================================================================================='''

    '''=============================================Random================================================================'''
    # overlap = np.isin(np.arange(len(IS.dl_tr.dataset)), gt_mislabelled_indices)
    # cum_overlap = np.cumsum(overlap)
    #
    # plt.plot(cum_overlap, label='Random')
    # plt.legend()
    # plt.tight_layout()
    # plt.gca().invert_xaxis()
    # plt.show()
    # plt.savefig('./images/mislabel_{}_{}_alltheta_noisylevel{}.pdf'.format(dataset_name, loss_type, noisy_level),
    #             bbox_inches='tight')
    '''============================================================================================================='''

    '''Relabelled data accuracy (only relabel harmful)'''
    '''Weighted KNN'''
    # start_time = time.time()
    # harmful_indices = np.load("{}/{}_{}_harmful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name,
    #                                                                                         IS.loss_type, 0, num_thetas,
    #                                                                                         noisy_level))
    # helpful_indices = np.load("{}/{}_{}_helpful_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name,
    #                                                                                         IS.loss_type, 0, num_thetas,
    #                                                                                         noisy_level))
    # influence_values = np.load(
    #     "{}/{}_{}_influence_values_testcls{}_SIF_theta{}_{}_step50.npy".format(basedir, IS.dataset_name, IS.loss_type,
    #                                                                            0, num_thetas, noisy_level))
    #
    # relabel_dict = {}
    # unique_labels, unique_counts = torch.unique(IS.train_label, return_counts=True)
    # median_shots_percls = unique_counts.median().item()
    # _, prob_relabel = kNN_label_pred(query_indices=harmful_indices,
    #                                  embeddings=IS.train_embedding,
    #                                  labels=IS.train_label,
    #                                  nb_classes=IS.dl_tr.dataset.nb_classes(),
    #                                  knn_k=median_shots_percls)
    # for kk in range(len(harmful_indices)):
    #     relabel_dict[harmful_indices[kk]] = prob_relabel[kk].detach().cpu().numpy()
    # print(time.time() - start_time)
    #
    # total_ct = 0
    # ct_correct = 0
    # for ind in gt_mislabelled_indices:
    #     if ind in relabel_dict.keys():
    #         total_ct += 1
    #         if relabel_dict[ind].argsort()[::-1][0] == IS.dl_tr.dataset.ys[ind]:
    #             if relabel_dict[ind].argsort()[::-1][1] == IS.dl_tr_clean.dataset.ys[ind]:
    #                 ct_correct += 1
    #         else:
    #             if relabel_dict[ind].argsort()[::-1][0] == IS.dl_tr_clean.dataset.ys[ind]:
    #                 ct_correct += 1
    #
    # print(ct_correct, total_ct, ct_correct/total_ct)

    '''Visualize'''
    # # first harmful then helpful
    training_sample_by_influence = influence_values.argsort()[::-1]
    overlap = np.isin(training_sample_by_influence, gt_mislabelled_indices)
    confusion_class_pairs = IS.get_confusion_class_pairs()
    training_sample_by_influence_in_mislabelled = [training_sample_by_influence[ct] for ct in range(len(training_sample_by_influence)) \
                                                   if overlap[ct] == True]
    #
    # # Get the two embeddings first
    # wrong_cls = confusion_class_pairs[0][0][0]  # FIXME: look at pairs associated with top-1 wrong class
    # confused_classes = [x[1] for x in confusion_class_pairs[0]]
    # wrong_indices, confused_indices = avg_confusion(wrong_cls, confused_classes,
    #                                                 IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
    # wrong_indices = np.concatenate(wrong_indices).ravel() # flatten
    # confused_indices = np.concatenate(confused_indices).ravel()
    #
    # # Display it
    # plt.rcParams.update({'axes.titlesize': 10, 'font.weight': 'bold'})
    # upper_bound = []
    # lower_bound = []
    #
    # fig, ax = plt.subplots(3, 11, figsize=(40, 40))
    # ax[0, 1].set_ylabel('Testing confusion samples', fontsize=10, rotation=0, labelpad=200, loc='center')
    # for ii in range(10):
    #     ax[0, ii+1].imshow(to_pil_image(read_image(IS.dl_ev.dataset.im_paths[wrong_indices[ii]])))
    #     ax[0, ii+1].set_title('Test Ind = {} \n Class = {}'.format(wrong_indices[ii], IS.dl_ev.dataset.ys[wrong_indices[ii]]))
    #     ax[0, ii+1].set_xticks(())
    #     ax[0, ii+1].set_yticks(())
    #     ax[0, ii+1].set_anchor('N')
    # ax[0, 0].set_axis_off()
    #
    # for it, harmful_ind in enumerate(training_sample_by_influence_in_mislabelled[:10]):
    #     harmful_img = to_pil_image(read_image(IS.dl_tr.dataset.im_paths[harmful_ind]))
    #     ax[1, it + 1].imshow(harmful_img)
    #     ax[1, it + 1].set_title('Class = {}'.format(IS.dl_tr.dataset.ys[harmful_ind]))
    #     ax[1, 1].set_ylabel('Top harmful influential',
    #                         fontsize=10, rotation=0, labelpad=200, loc='center')
    #     ax[1, it + 1].set_xticks(())
    #     ax[1, it + 1].set_yticks(())
    #     ax[1, it + 1].set_anchor('N')
    #
    # ax[1, 0].set_axis_off()
    #
    # for it, helpful_ind in enumerate(training_sample_by_influence_in_mislabelled[::-1][:10]):
    #     helpful_img = to_pil_image(read_image(IS.dl_tr.dataset.im_paths[helpful_ind]))
    #     ax[2, it + 1].imshow(helpful_img)
    #     ax[2, it + 1].set_title('Class = {}'.format(IS.dl_tr.dataset.ys[helpful_ind]))
    #     ax[2, 1].set_ylabel('Top helpful influential',
    #                         fontsize=10, rotation=0, labelpad=200, loc='center')
    #     ax[2, it + 1].set_xticks(())
    #     ax[2, it + 1].set_yticks(())
    #     ax[2, it + 1].set_anchor('N')
    # ax[2, 0].set_axis_off()
    #
    # plt.show()

    '''================================upweight helpful and see=================================================='''
    '''Other: get confusion (before VS after)'''
    # IS.model = IS._load_model()  # reload the original weights
    # test_features = IS.get_test_features()
    # confusion_class_pairs = IS.get_confusion_class_pairs()
    #
    # wrong_cls = confusion_class_pairs[0][0][0]
    # confused_classes = [x[1] for x in confusion_class_pairs[0]]
    #
    # IS.model = IS._load_model() # reload the original weights
    # inter_dist_orig, _ = grad_confusion(IS.model, test_features, wrong_cls, confused_classes,
    #                                     IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
    # print("Original d(G_p): ", inter_dist_orig)
    #
    # # reload weights as new
    # weight_path = 'models/dvi_data_{}_{}_loss{}_2_1/ResNet_512_Model/Epoch_1/{}_{}_trainval_{}_{}.pth'.format(
    #                  dataset_name, seed,
    #                 '{}_EIF'.format(loss_type),
    #                  dataset_name, dataset_name, 512, seed)
    #
    # IS.model.load_state_dict(torch.load(weight_path))
    # inter_dist_after, _ = grad_confusion(IS.model, test_features, wrong_cls, confused_classes,
    #                                      IS.testing_nn_label, IS.testing_label, IS.testing_nn_indices)
    # print("After d(G_p) EIF: ", inter_dist_after)


    '''================================look at how many helpful noisy are ignored by the model=================================================='''
    nn_indices, nn_labels = assign_by_euclidian_at_k_indices(IS.train_embedding, IS.dl_tr_clean.dataset.ys, 5)  # predict nn labels

    helpful_noisy_indices = np.asarray(training_sample_by_influence_in_mislabelled[::-1][:100])
    harmful_noisy_indices = np.asarray(training_sample_by_influence_in_mislabelled[:100])

    perc_nn_are_orig_class_helpfulnoisy = np.equal(nn_labels[helpful_noisy_indices, :],
                                     np.asarray(IS.dl_tr_clean.dataset.ys)[helpful_noisy_indices][:, np.newaxis]).sum(-1) / nn_labels.shape[-1]
    perc_nn_are_orig_class_harmfulnoisy = np.equal(nn_labels[harmful_noisy_indices, :],
                                     np.asarray(IS.dl_tr_clean.dataset.ys)[harmful_noisy_indices][:, np.newaxis]).sum(-1) / nn_labels.shape[-1]
    print('Avg %top-5 nn belongs to original class label for helpful noisy = {}'.format(np.mean(perc_nn_are_orig_class_helpfulnoisy)))
    print('Avg %top-5 nn belongs to original class label for harmful noisy = {}'.format(np.mean(perc_nn_are_orig_class_harmfulnoisy)))
    print('haha')

    '''================================take a random step and observe whether helpful are still helpful=================================================='''
    confusion_class_pairs = IS.get_confusion_class_pairs()[0]
    all_features = IS.get_test_features()  # test features (N, 2048)
    theta_orig = IS.model.module[-1].weight.data  # original theta
    deltaL_deltaD = []
    theta_list = torch.tensor([])

    wrong_cls = confusion_class_pairs[0][0]  # FIXME: look at pairs associated with top-1 wrong class
    confused_classes = [x[1] for x in confusion_class_pairs]
    inter_dist_orig, _ = grad_confusion(IS.model, all_features, wrong_cls, confused_classes,
                                        IS.testing_nn_label, IS.testing_label,
                                        IS.testing_nn_indices)  # original D

    theta_new = IS.theta_for_groups_confusion_gettheta(all_features,
                                                        wrong_cls, confused_classes,
                                                        steps=50,
                                                        descent=False)
    theta_list = torch.cat([theta_list, theta_new.detach().cpu().unsqueeze(0)], dim=0)
    theta_new = IS.theta_for_groups_confusion_gettheta(all_features,
                                                       wrong_cls, confused_classes,
                                                       steps=50,
                                                       descent=True)
    theta_list = torch.cat([theta_list, theta_new.detach().cpu().unsqueeze(0)], dim=0)

    deltaD, deltaL, _ = IS.get_theta_orthogonalization_forclasses(prev_thetas=theta_list,
                                                            all_features=all_features,
                                                            wrong_cls=wrong_cls,
                                                            confuse_classes=confused_classes,
                                                            theta_orig=theta_orig,
                                                            inter_dist_orig=inter_dist_orig
                                                            )

    mean_deltaL_deltaD = deltaL * deltaD
    influence_values_random = np.asarray(mean_deltaL_deltaD)
    print(np.isin(helpful_noisy_indices, np.where(influence_values_random < 0)[0]))
    print(np.sum(np.isin(helpful_noisy_indices, np.where(influence_values_random < 0)[0])))
    print(np.sum(np.isin(harmful_noisy_indices, np.where(influence_values_random < 0)[0])))
    print('haha')
