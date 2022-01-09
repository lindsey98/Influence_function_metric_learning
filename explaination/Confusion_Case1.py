
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from PIL import Image
from Influence_function.influential_sample import InfluentialSample
from explaination.CAM_methods import *
from Influence_function.influence_function import *
from explaination.background_removal import remove_background
import utils
import dataset
from torchvision import transforms
from dataset.utils import RGBAToRGB, ScaleIntensities
from utils import overlay_mask
from utils import predict_batchwise_debug
from evaluation import assign_by_euclidian_at_k_indices, assign_diff_cls_neighbor, assign_same_cls_neighbor
import sklearn
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class DistinguishFeat(InfluentialSample):
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 test_crop=False, sz_embedding=512, epoch=40):

        super().__init__(dataset_name, seed, loss_type, config_name,
                          test_crop, sz_embedding, epoch)

    def GradAnalysis(self,
                     wrong_indices, confuse_indices, wrong_samecls_indices,
                     dl, base_dir='Grad_Test'):
        '''Visualize all confusion pairs'''
        assert len(wrong_indices) == len(confuse_indices)
        assert len(wrong_indices) == len(wrong_samecls_indices)

        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
        model_copy = self._load_model()
        model_copy.eval()

        for ind1, ind2, ind3 in zip(wrong_indices, confuse_indices, wrong_samecls_indices):
            cam_extractor1 = GradCAMCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer
            cam_extractor2 = GradCAMCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer

            # Get the two embeddings first
            img1 = to_pil_image(read_image(dl.dataset.im_paths[ind1]))
            img2 = to_pil_image(read_image(dl.dataset.im_paths[ind2]))
            img3 = to_pil_image(read_image(dl.dataset.im_paths[ind3]))

            cam_extractor1._hooks_enabled = True
            model_copy.zero_grad()
            emb1 = model_copy(dl.dataset.__getitem__(ind1)[0].unsqueeze(0).cuda())
            emb2 = model_copy(dl.dataset.__getitem__(ind2)[0].unsqueeze(0).cuda())
            activation_map2 = cam_extractor1(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
            result2, _ = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)

            cam_extractor2._hooks_enabled = True
            model_copy.zero_grad()
            emb1 = model_copy(dl.dataset.__getitem__(ind1)[0].unsqueeze(0).cuda())
            emb3 = model_copy(dl.dataset.__getitem__(ind3)[0].unsqueeze(0).cuda())
            activation_map3 = cam_extractor2(torch.dot(emb1.detach().squeeze(0), emb3.squeeze(0)))
            result3, _ = overlay_mask(img3, to_pil_image(activation_map3[0].detach().cpu(), mode='F'), alpha=0.5)

            # Display it
            fig = plt.figure()
            fig.subplots_adjust(top=0.8)

            ax = fig.add_subplot(2, 3, 1)
            ax.imshow(img1)
            ax.title.set_text('Ind = {}, Class = {}'.format(ind1, dl.dataset.ys[ind1]))
            plt.axis('off')

            ax = fig.add_subplot(2, 3, 2)
            ax.imshow(img2)
            ax.title.set_text('Ind = {}, Class = {}'.format(ind2, dl.dataset.ys[ind2]))
            plt.axis('off')

            ax = fig.add_subplot(2, 3, 3)
            ax.imshow(img3)
            ax.title.set_text('Ind = {}, Class = {}'.format(ind3, dl.dataset.ys[ind3]))
            plt.axis('off')

            ax=fig.add_subplot(2, 3, 5)
            ax.imshow(result2)
            plt.axis('off')

            ax=fig.add_subplot(2, 3, 6)
            ax.imshow(result3)
            plt.axis('off')
            # plt.show()
            plt.savefig('./{}/{}/{}_{}.png'.format(base_dir, self.dataset_name,
                                                   ind1, ind2))
            plt.close()

    def VisTrainNN(self, pair_ind1, pair_ind2,
                   anchor_ind, orig_NN_ind, orig_same_cls_NN_ind,
                   model,
                   dl,
                   base_dir):

        plt_dir = './{}/{}/{}_{}'.format(base_dir, self.dataset_name, pair_ind1, pair_ind2)
        os.makedirs(plt_dir, exist_ok=True)
        model.eval()

        fig = plt.figure()
        fig.subplots_adjust(top=0.8)

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[anchor_ind])))
        ax.title.set_text('Ind:{}, Class:{}'.format(anchor_ind, dl.dataset.ys[anchor_ind]))
        plt.axis('off')

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_NN_ind])))

        model.zero_grad()
        emb1 = model(dl.dataset.__getitem__(anchor_ind)[0].unsqueeze(0).cuda())
        emb2 = model(dl.dataset.__getitem__(orig_NN_ind)[0].unsqueeze(0).cuda())
        d = calc_inter_dist_pair(emb1, emb2)

        ax.title.set_size(7)
        ax.title.set_text('Class:{}, D:{:.2f}'.format(dl.dataset.ys[orig_NN_ind], d))
        plt.axis('off')

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_same_cls_NN_ind])))

        model.zero_grad()
        emb1 = model(dl.dataset.__getitem__(anchor_ind)[0].unsqueeze(0).cuda())
        emb2 = model(dl.dataset.__getitem__(orig_same_cls_NN_ind)[0].unsqueeze(0).cuda())
        d = calc_inter_dist_pair(emb1, emb2)

        ax.title.set_size(7)
        ax.title.set_text('Class:{}, D:{:.2f}'.format(dl.dataset.ys[orig_same_cls_NN_ind], d))
        plt.axis('off')

        # plt.show()
        plt.savefig('./{}/{}.png'.format(plt_dir, anchor_ind))
        plt.close()

    # def VisTrain(self, wrong_ind, confusion_ind,
    #              interest_indices, orig_NN_indices, curr_NN_indices,
    #              model1, model2,
    #              dl,
    #              base_dir):
    #
    #     plt_dir = './{}/{}/{}_{}'.format(base_dir, self.dataset_name, wrong_ind, confusion_ind)
    #     os.makedirs(plt_dir, exist_ok=True)
    #     model1.eval(); model2.eval()
    #
    #     for kk, ind in enumerate(interest_indices):
    #         fig = plt.figure()
    #         fig.subplots_adjust(top=0.8)
    #
    #         ax = fig.add_subplot(3, 3, 1)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[ind])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(ind, dl.dataset.ys[ind]))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(3, 3, 2)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(orig_NN_indices[kk], dl.dataset.ys[orig_NN_indices[kk]]))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(3, 3, 3)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(curr_NN_indices[kk], dl.dataset.ys[curr_NN_indices[kk]]))
    #         plt.axis('off')
    #
    #         cam_extractor1 = GradCAMCustomize(model1,
    #                                           target_layer=model1.module[0].base.layer4)  # to last layer
    #         cam_extractor2 = GradCAMCustomize(model2,
    #                                           target_layer=model2.module[0].base.layer4)  # to last layer
    #         cam_extractor1._hooks_enabled = True
    #         model1.zero_grad()
    #         emb1 = model1(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         emb2 = model1(dl.dataset.__getitem__(orig_NN_indices[kk])[0].unsqueeze(0).cuda())
    #         activation_map1 = cam_extractor1(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
    #         img1 = to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]]))
    #         result1, _ = overlay_mask(img1, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)
    #         emb1 = F.normalize(emb1, p=2, dim=-1)
    #         emb2 = F.normalize(emb2, p=2, dim=-1)
    #         d1 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()
    #
    #         cam_extractor2._hooks_enabled = True
    #         model2.zero_grad()
    #         emb1 = model2(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         emb2 = model2(dl.dataset.__getitem__(curr_NN_indices[kk])[0].unsqueeze(0).cuda())
    #         activation_map2 = cam_extractor2(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
    #         img2 = to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]]))
    #         result2, _ = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)
    #         emb1 = F.normalize(emb1, p=2, dim=-1)
    #         emb2 = F.normalize(emb2, p=2, dim=-1)
    #         d2 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()
    #
    #         ax = fig.add_subplot(3, 3, 5)
    #         ax.imshow(result1)
    #         ax.title.set_text('Distance = {:.4f}'.format(d1))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(3, 3, 6)
    #         ax.imshow(result2)
    #         ax.title.set_text('Distance = {:.4f}'.format(d2))
    #         plt.axis('off')
    #
    #         # Affinity to proxy
    #         proxy = self.criterion.proxies[dl.dataset.ys[ind]].detach()
    #         cam_extractor1 = GradCAMCustomize(model1,
    #                                           target_layer=model1.module[0].base.layer4)  # to last layer
    #         cam_extractor2 = GradCAMCustomize(model2,
    #                                           target_layer=model2.module[0].base.layer4)  # to last layer
    #         cam_extractor1._hooks_enabled = True
    #         model1.zero_grad()
    #         emb1 = model1(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         activation_map1 = cam_extractor1(torch.dot(emb1.squeeze(0), proxy.to(emb1.device)))
    #         img = to_pil_image(read_image(dl.dataset.im_paths[ind]))
    #         result1, _ = overlay_mask(img, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)
    #
    #         cam_extractor2._hooks_enabled = True
    #         model2.zero_grad()
    #         emb1 = model2(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         activation_map2 = cam_extractor2(torch.dot(emb1.squeeze(0), proxy.to(emb1.device)))
    #         result2, _ = overlay_mask(img, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)
    #
    #         ax = fig.add_subplot(3, 3, 8)
    #         ax.imshow(result1)
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(3, 3, 9)
    #         ax.imshow(result2)
    #         plt.axis('off')
    #         # plt.show()
    #         plt.savefig('./{}/{}.png'.format(plt_dir, ind))
    #         plt.close()


if __name__ == '__main__':

    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    test_crop = False

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, test_crop)

    '''Analyze confusing features for all confusion classes'''
    '''Step 1: Visualize all pairs (confuse), Find interesting pairs'''
    testing_embedding, testing_label = DF.testing_embedding, DF.testing_label
    test_nn_indices, test_nn_label = assign_by_euclidian_at_k_indices(testing_embedding, testing_label, 1)
    test_nn_indices, test_nn_label = test_nn_indices.flatten(), test_nn_label.flatten()
    wrong_indices = (test_nn_label != testing_label.detach().cpu().numpy().flatten()).nonzero()[0]
    confuse_indices = test_nn_indices[wrong_indices]
    print(len(confuse_indices))
    assert len(wrong_indices) == len(confuse_indices)

    # Same class 1st NN
    distances = sklearn.metrics.pairwise.pairwise_distances(testing_embedding)  # (N_train, N_train)
    diff_cls_mask = (testing_label[:, None] != testing_label).detach().cpu().numpy().nonzero()
    distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
    test_nn_indices_same_cls = np.argsort(distances, axis=1)[:, 1]
    wrong_test_nn_indices_same_cls = test_nn_indices_same_cls[wrong_indices]
    assert len(wrong_indices) == len(wrong_test_nn_indices_same_cls)

    # DF.GradAnalysis( wrong_indices, confuse_indices, wrong_test_nn_indices_same_cls,
    #                  DF.dl_ev, base_dir='Confuse_Vis')
    # exit()

    '''Step 2: Identify influential training points for a specific pair'''
    pair_ind1 = 35; pair_ind2 = 2551
    base_dir = 'Confuse_pair_influential_data/{}'.format(DF.dataset_name)
    os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists('./{}/All_influence_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2)):
        all_features = DF.get_features()
        # sanity check: IS.viz_2sample(IS.dl_ev, wrong_ind, confuse_ind)
        training_sample_by_influence, influence_values = DF.single_influence_func(all_features, [pair_ind1], [pair_ind2])
        helpful_indices = np.where(influence_values < 0)[0]
        harmful_indices = np.where(influence_values > 0)[0]
        np.save('./{}/Allhelpful_indices_{}_{}'.format(base_dir, pair_ind1, pair_ind2), helpful_indices)
        np.save('./{}/Allharmful_indices_{}_{}'.format(base_dir, pair_ind1, pair_ind2), harmful_indices)
        np.save('./{}/All_influence_{}_{}'.format(base_dir, pair_ind1, pair_ind2), influence_values)
    else:
        helpful_indices = np.load('./{}/Allhelpful_indices_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2))
        harmful_indices = np.load('./{}/Allharmful_indices_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2))
        influence_values = np.load('./{}/All_influence_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2))

    '''Step 3: Visualize those training'''
    # Global 1st NN
    training_embedding, training_label = DF.train_embedding, DF.train_label
    train_nn_indices, train_nn_label = assign_by_euclidian_at_k_indices(training_embedding, training_label, 1)
    train_nn_indices, train_nn_label = train_nn_indices.flatten(), train_nn_label.flatten()

    # Same class 1st NN
    distances = sklearn.metrics.pairwise.pairwise_distances(training_embedding)  # (N_train, N_train)
    diff_cls_mask = (training_label[:, None] != training_label).detach().cpu().numpy().nonzero()
    distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
    train_nn_indices_same_cls = np.argsort(distances, axis=1)[:, 1]

    assert len(train_nn_indices_same_cls) == len(train_nn_indices)
    assert len(training_label) == len(train_nn_indices)

    # print([influence_values[ind] for ind in influence_values.argsort()[-50:]])
    # print([DF.dl_tr.dataset.ys[ind] for ind in influence_values.argsort()[-50:]])

    # DF.model = DF._load_model()  # reload the original weights
    # for harmful in influence_values.argsort()[-20:]:
    #     DF.VisTrainNN(pair_ind1, pair_ind2,
    #                   harmful,
    #                   train_nn_indices[harmful],
    #                   train_nn_indices_same_cls[harmful],
    #                   DF.model,
    #                   DF.dl_tr,
    #                   base_dir='ModelS_HumanD')

    '''Step 4: Save harmful indices as well as its neighboring indices'''
    # top_harmful_indices = influence_values.argsort()[-50:]
    # top_harmful_nn_indices = train_nn_indices[top_harmful_indices]
    # top_harmful_nn_samecls_indices = train_nn_indices_same_cls[top_harmful_indices]
    #
    # relabel_dict = {}
    # for kk in range(len(top_harmful_indices)):
    #     if DF.dl_tr.dataset.ys[top_harmful_nn_indices[kk]] != DF.dl_tr.dataset.ys[top_harmful_nn_samecls_indices[kk]]: # inconsistent label between global NN and same class NN
    #         relabel_dict[top_harmful_indices[kk]] = [DF.dl_tr.dataset.ys[top_harmful_nn_samecls_indices[kk]],
    #                                                  DF.dl_tr.dataset.ys[top_harmful_nn_indices[kk]]]
    #
    #
    # with open('./{}/relabeldict_{}_{}.pkl'.format(base_dir, pair_ind1, pair_ind2), 'wb') as handle:
    #     pickle.dump(relabel_dict, handle)

    '''Step 5: Verify that the model after training is better?'''
    DF.model = DF._load_model()  # reload the original weights
    new_features = DF.get_features()

    inter_dist_orig, _ = grad_confusion_pair(DF.model, new_features, [pair_ind1], [pair_ind2])
    print('Original distance: ', inter_dist_orig)

    new_weight_path = 'models/dvi_data_{}_{}_loss{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
        dataset_name,
        seed,
        'ProxyNCA_pfix_softlabel_{}_{}'.format(pair_ind1, pair_ind2),
        1, dataset_name,
        dataset_name,
        512, seed)  # reload weights as new
    DF.model.load_state_dict(torch.load(new_weight_path))
    inter_dist_after, _ = grad_confusion_pair(DF.model, new_features, [pair_ind1], [pair_ind2])
    print('After distance: ', inter_dist_orig)


