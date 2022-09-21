from Influence_function.influence_function import OrigIF, EIF
from Influence_function.EIF_utils import grad_confusion
import os
from Influence_function.IF_utils import *
import numpy as np
from evaluation import assign_by_euclidian_at_k_indices
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == '__main__':
    sz_embedding = 512; epoch = 40; test_crop = False; topk_cls = 30
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cub';  config_name = 'cub_ProxyNCA_prob_orig'; seed = 0
    # loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'cars'; config_name = 'cars_ProxyNCA_prob_orig'; seed = 3
    loss_type = 'ProxyNCA_prob_orig'; dataset_name = 'inshop'; config_name = 'inshop_ProxyNCA_prob_orig'; seed = 4

    IS_IF = OrigIF(dataset_name, seed, loss_type, config_name, 'dataset/config.json', test_crop, sz_embedding, epoch, 'ResNet')
    IS_EIF = EIF(dataset_name, seed, loss_type, config_name, 'dataset/config.json', test_crop, sz_embedding, epoch, 'ResNet')

    '''Given a confusion pair, find its top 5 harmful training identified by EIF and IF'''
    '''Step 1: Get all wrong pairs'''
    testing_embedding, testing_label = IS_IF.testing_embedding, IS_IF.testing_label
    test_nn_indices, test_nn_label = assign_by_euclidian_at_k_indices(testing_embedding, testing_label, 1)
    wrong_indices = (test_nn_label.flatten() != testing_label.detach().cpu().numpy().flatten()).nonzero()[0]
    confuse_indices = test_nn_indices.flatten()[wrong_indices]
    print(len(confuse_indices))
    assert len(wrong_indices) == len(confuse_indices)


    for kk in range(50):
        wrong_ind = wrong_indices[kk]
        confuse_ind = confuse_indices[kk]

        # IF
        try:
            influence_values_IF = np.load(
                './{}/{}_influence_values_{}_{}.npy'.format(
                    'Confuse_pair_influential_data_baselines/{}'.format(IS_IF.dataset_name),
                    IS_IF.loss_type, wrong_ind, confuse_ind))
        except FileNotFoundError:
            train_features = IS_IF.get_train_features()
            test_features = IS_IF.get_test_features()  # (N, 2048)
            influence_values_IF = IS_IF.IF_for_pairs_confusion(train_features=train_features, test_features=test_features,
                                                               wrong_indices=[wrong_ind], confuse_indices=[confuse_ind])
            influence_values_IF = np.asarray(influence_values_IF).flatten()
            np.save('./{}/{}_influence_values_{}_{}'.format(
                    'Confuse_pair_influential_data_baselines/{}'.format(IS_IF.dataset_name),
                    IS_IF.loss_type, wrong_ind, confuse_ind), influence_values_IF)

        top_5_harmful_indices_IF = influence_values_IF.argsort()[:5]

        # EIF
        try:
            influence_values_EIF = np.load(
                './{}/{}_influence_values_{}_{}.npy'.format(
                    'Confuse_pair_influential_data/{}'.format(IS_EIF.dataset_name),
                    IS_EIF.loss_type, wrong_ind, confuse_ind))
        except FileNotFoundError:
            mean_deltaL_deltaD_EIF = IS_EIF.EIF_for_pairs_confusion([wrong_ind, confuse_ind], num_thetas=1, steps=50)
            influence_values_EIF = np.asarray(mean_deltaL_deltaD_EIF)
            np.save('./{}/{}_influence_values_{}_{}'.format(
                    'Confuse_pair_influential_data/{}'.format(IS_EIF.dataset_name),
                    IS_EIF.loss_type, wrong_ind, confuse_ind), influence_values_EIF)

        top_5_harmful_indices_EIF = influence_values_EIF.argsort()[::-1][:5]

        print(top_5_harmful_indices_IF)
        print(top_5_harmful_indices_EIF)

        # Get the two embeddings first
        img1 = to_pil_image(read_image(IS_IF.dl_ev.dataset.im_paths[wrong_ind]))
        img2 = to_pil_image(read_image(IS_IF.dl_ev.dataset.im_paths[confuse_ind]))
        # Display it
        plt.rcParams.update({'axes.titlesize': 40, 'font.weight': 'bold'})
        upper_bound = []
        lower_bound = []

        fig, ax = plt.subplots(3, 6, figsize=(40, 40))
        ax[0, 0].set_axis_off(); ax[0, 1].set_axis_off()
        ax[0, 3].set_axis_off(); ax[0, 5].set_axis_off()

        ax[0, 2].imshow(img1)
        ax[0, 2].set_title('Test Ind = {} \n Class = {}'.format(wrong_ind, IS_IF.dl_ev.dataset.ys[wrong_ind]))
        ax[0, 2].set_xticks(())
        ax[0, 2].set_yticks(())
        ax[0, 2].set_ylabel('Testing confusion pair', fontsize=50, rotation=0, labelpad=300, loc='center')
        ax[0, 2].set_anchor('N')

        ax[0, 4].imshow(img2)
        ax[0, 4].set_title('Test Ind = {} \n Class = {}'.format(confuse_ind, IS_IF.dl_ev.dataset.ys[confuse_ind]))
        ax[0, 4].set_xticks(())
        ax[0, 4].set_yticks(())
        ax[0, 4].set_anchor('N')

        for it, harmful_ind in enumerate(top_5_harmful_indices_IF):
            harmful_img = to_pil_image(read_image(IS_IF.dl_tr.dataset.im_paths[harmful_ind]))
            ax[1, it+1].imshow(harmful_img)
            ax[1, it+1].set_title('Class = {}'.format(IS_IF.dl_tr.dataset.ys[harmful_ind]))
            ax[1, 1].set_ylabel('Top harmful by \n IF',
                                fontsize=40, rotation=0, labelpad=200, loc='center')
            ax[1, it+1].set_xticks(())
            ax[1, it+1].set_yticks(())
            ax[1, it+1].set_anchor('N')

        ax[1, 0].set_axis_off()

        for it, harmful_ind in enumerate(top_5_harmful_indices_EIF):
            harmful_img = to_pil_image(read_image(IS_EIF.dl_tr.dataset.im_paths[harmful_ind]))
            ax[2, it+1].imshow(harmful_img)
            ax[2, it+1].set_title('Class = {}'.format(IS_EIF.dl_tr.dataset.ys[harmful_ind]))
            ax[2, 1].set_ylabel('Top harmful by \n EIF',
                                fontsize=40, rotation=0, labelpad=200, loc='center')
            ax[2, it+1].set_xticks(())
            ax[2, it+1].set_yticks(())
            ax[2, it+1].set_anchor('N')
        ax[2, 0].set_axis_off()

        for y in [0.66, 0.45]:
            line = plt.Line2D([0.1, 1.0], [y, y],
                              color="green", linestyle='dashed', linewidth=2)
            fig.add_artist(line)

        plt.subplots_adjust(wspace=0.05, hspace=0.0)
        plt.savefig('./{}/{}/{}_{}.png'.format('Grad_Test', IS_IF.dataset_name, wrong_ind, confuse_ind), bbox_inches='tight')
        # plt.show()
        # break