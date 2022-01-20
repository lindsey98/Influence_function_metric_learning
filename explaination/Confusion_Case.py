
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
import utils
import dataset
from torchvision import transforms
from dataset.utils import RGBAToRGB, ScaleIntensities
from utils import overlay_mask
from utils import predict_batchwise_debug
from evaluation import assign_by_euclidian_at_k_indices, assign_diff_cls_neighbor, assign_same_cls_neighbor
import sklearn
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class SampleRelabel(InfluentialSample):
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

    def calc_relabel_dict(self, lookat_harmful, relabel_method,
                          harmful_indices, helpful_indices, train_nn_indices, train_nn_indices_same_cls,
                          base_dir, pair_ind1, pair_ind2):

        assert isinstance(lookat_harmful, bool)
        assert relabel_method in ['hard', 'soft']
        if lookat_harmful:
            top_harmful_indices = harmful_indices  # top_harmful_indices = influence_values.argsort()[-50:]
            top_harmful_nn_indices = train_nn_indices[top_harmful_indices]
            top_harmful_nn_samecls_indices = train_nn_indices_same_cls[top_harmful_indices]

            if relabel_method == 'hard':
                relabel_dict = {}
                for kk in range(len(top_harmful_indices)):
                    if self.dl_tr.dataset.ys[top_harmful_nn_indices[kk]] != self.dl_tr.dataset.ys[top_harmful_nn_samecls_indices[kk]]: # inconsistent label between global NN and same class NN
                        relabel_dict[top_harmful_indices[kk]] = [self.dl_tr.dataset.ys[top_harmful_nn_samecls_indices[kk]],
                                                                 self.dl_tr.dataset.ys[top_harmful_nn_indices[kk]]]
                with open('./{}/Allrelabeldict_{}_{}.pkl'.format(base_dir, pair_ind1, pair_ind2), 'wb') as handle:
                    pickle.dump(relabel_dict, handle)

            elif relabel_method == 'soft':
                relabel_dict = {}
                unique_labels, unique_counts = torch.unique(self.train_label, return_counts=True)
                median_shots_percls = unique_counts.median().item()
                _, pred_scores = kNN_label_pred(top_harmful_indices, self.train_embedding, self.train_label,
                                                self.dl_tr.dataset.nb_classes(), median_shots_percls)
                for kk in range(len(top_harmful_indices)):
                    relabel_dict[top_harmful_indices[kk]] = pred_scores[kk]
                with open('./{}/Allrelabeldict_{}_{}_soft.pkl'.format(base_dir, pair_ind1, pair_ind2), 'wb') as handle:
                    pickle.dump(relabel_dict, handle)

            else:
                raise NotImplemented

        else:
            top_helpful_indices = helpful_indices
            top_helpful_nn_indices = train_nn_indices[top_helpful_indices]
            top_helpful_nn_samecls_indices = train_nn_indices_same_cls[top_helpful_indices]

            if relabel_method == 'hard':
                relabel_dict = {}
                for kk in range(len(top_helpful_indices)):
                    if DF.dl_tr.dataset.ys[top_helpful_nn_indices[kk]] != DF.dl_tr.dataset.ys[top_helpful_nn_samecls_indices[kk]]: # inconsistent label between global NN and same class NN
                        relabel_dict[top_helpful_indices[kk]] = [DF.dl_tr.dataset.ys[top_helpful_nn_samecls_indices[kk]],
                                                                 DF.dl_tr.dataset.ys[top_helpful_nn_indices[kk]]]
                with open('./{}/Allrelabeldict_{}_{}.pkl'.format(base_dir, pair_ind1, pair_ind2), 'wb') as handle:
                    pickle.dump(relabel_dict, handle)

            elif relabel_method == 'soft':
                relabel_dict = {}
                unique_labels, unique_counts = torch.unique(self.train_label, return_counts=True)
                median_shots_percls = unique_counts.median().item()
                _, pred_scores = kNN_label_pred(top_helpful_indices, self.train_embedding,
                                                self.train_label,
                                                self.dl_tr.dataset.nb_classes(), median_shots_percls)

                for kk in range(len(top_helpful_indices)):
                    relabel_dict[top_helpful_indices[kk]] = pred_scores[kk]

                with open('./{}/Allrelabeldict_{}_{}_soft.pkl'.format(base_dir, pair_ind1, pair_ind2), 'wb') as handle:
                    pickle.dump(relabel_dict, handle)

            else:
                raise NotImplemented


    def getNN_indices(self, embedding, label):

        # global 1st NN
        nn_indices, nn_label = assign_by_euclidian_at_k_indices(embedding, label, 1)
        nn_indices, nn_label = nn_indices.flatten(), nn_label.flatten()

        # Same class 1st NN
        distances = sklearn.metrics.pairwise.pairwise_distances(embedding)  # (N_train, N_train)
        diff_cls_mask = (label[:, None] != label).detach().cpu().numpy().nonzero()
        distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
        nn_indices_same_cls = np.argsort(distances, axis=1)[:, 1]

        return nn_indices, nn_label, nn_indices_same_cls


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR, https://github.com/PatrickHua/SimSiam/blob/main/tools/knn_monitor.py
def kNN_label_pred(query_indices, embeddings, labels, nb_classes, knn_k):

    distances = sklearn.metrics.pairwise.pairwise_distances(embeddings)
    indices = np.argsort(distances, axis=1)[:, 1: knn_k + 1]
    query_nn_indices = indices[query_indices] # (B, knn_k)
    query_nn_labels = torch.gather(labels.expand(query_indices.shape[0], -1),
                                   dim=-1,
                                   index=torch.from_numpy(query_nn_indices)) # (B, knn_k)

    query2nn_dist = distances[np.repeat(query_indices, knn_k), query_nn_indices.flatten()] # (B*knn_k, )
    query2nn_dist = query2nn_dist.reshape(len(query_indices), -1) # (B, knn_k)
    query2nn_dist_exp = (-torch.from_numpy(query2nn_dist)).exp() # (B, knn_k)

    one_hot_label = torch.zeros(query_indices.shape[0] * knn_k, nb_classes, device=query_nn_labels.device) # (B*knn_k, C)
    one_hot_label = one_hot_label.scatter(dim=-1, index=query_nn_labels.view(-1, 1).type(torch.int64), value=1.0)

    raw_pred_scores = torch.sum(one_hot_label.view(query_indices.shape[0], -1, nb_classes) * query2nn_dist_exp.unsqueeze(dim=-1), dim=1) # (B, C)
    pred_scores = raw_pred_scores / torch.sum(raw_pred_scores, dim=-1, keepdim=True) # (B, C)
    pred_labels = torch.argsort(pred_scores, dim=-1, descending=True) # (B, C)
    return pred_labels, pred_scores

if __name__ == '__main__':

    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub';  config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4
    # dataset_name = 'sop'; config_name = 'sop'; seed = 3

    DF = SampleRelabel(dataset_name, seed, loss_type, config_name, test_crop)

    '''Analyze confusing features for all confusion classes'''
    '''Step 1: Visualize all pairs (confuse), Find interesting pairs'''
    # test_nn_indices, test_nn_label, test_nn_indices_same_cls = DF.getNN_indices(DF.testing_embedding, DF.testing_label)
    # wrong_indices = (test_nn_label != DF.testing_label.detach().cpu().numpy().flatten()).nonzero()[0]
    # confuse_indices = test_nn_indices[wrong_indices]
    # print(len(confuse_indices))
    # assert len(wrong_indices) == len(confuse_indices)
    #
    # # Same class 1st NN
    # wrong_test_nn_indices_same_cls = test_nn_indices_same_cls[wrong_indices]
    # assert len(wrong_indices) == len(wrong_test_nn_indices_same_cls)
    #
    # DF.GradAnalysis( wrong_indices, confuse_indices, wrong_test_nn_indices_same_cls,
    #                  DF.dl_ev, base_dir='Confuse_Vis')
    # exit()

    '''Step 2: Identify influential training points for a specific pair'''
    pair_ind1, pair_ind2 = 558, 522
    lookat_harmful = False
    base_dir = 'Confuse_pair_influential_data/{}'.format(DF.dataset_name)
    os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists('./{}/All_influence_{}_{}.npy'.format(base_dir, pair_ind1, pair_ind2)):
        all_features = DF.get_features()
        # sanity check:
        DF.viz_2sample(DF.dl_ev, pair_ind1, pair_ind2)
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

    # '''Step 3: Visualize those training'''
    # # Global 1st NN
    train_nn_indices, train_nn_label, train_nn_indices_same_cls = DF.getNN_indices(DF.train_embedding, DF.train_label)
    assert len(train_nn_indices_same_cls) == len(train_nn_indices)
    assert len(DF.train_label) == len(train_nn_indices)
    #
    # DF.model = DF._load_model()  # reload the original weights
    # # for harmful in influence_values.argsort()[-20:]:
    # #     DF.VisTrainNN(pair_ind1, pair_ind2,
    # #                   harmful,
    # #                   train_nn_indices[harmful],
    # #                   train_nn_indices_same_cls[harmful],
    # #                   DF.model,
    # #                   DF.dl_tr,
    # #                   base_dir='ModelS_HumanD')
    #
    # # for helpful in influence_values.argsort()[:20]:
    # #     DF.VisTrainNN(pair_ind1, pair_ind2,
    # #                   helpful,
    # #                   train_nn_indices[helpful],
    # #                   train_nn_indices_same_cls[helpful],
    # #                   DF.model,
    # #                   DF.dl_tr,
    # #                   base_dir='ModelD_HumanS')

    '''Step 4: Save harmful indices as well as its neighboring indices'''
    DF.calc_relabel_dict(lookat_harmful=lookat_harmful, relabel_method='soft',
                          harmful_indices=harmful_indices, helpful_indices=helpful_indices,
                          train_nn_indices=train_nn_indices, train_nn_indices_same_cls=train_nn_indices_same_cls,
                          base_dir=base_dir, pair_ind1=pair_ind1, pair_ind2=pair_ind2)
    exit()

    '''Step 5: Verify that the model after training is better?'''
    DF.model = DF._load_model()  # reload the original weights
    new_features = DF.get_features()
    inter_dist_orig, _ = grad_confusion_pair(DF.model, new_features, [pair_ind1], [pair_ind2])
    print('Original distance: ', inter_dist_orig)

    new_weight_path = 'models/dvi_data_{}_{}_loss{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
        dataset_name,
        seed,
        'ProxyNCA_pfix_softlabel_{}_{}_continue_soft'.format(pair_ind1, pair_ind2),
        5, dataset_name,
        dataset_name,
        512, seed)  # reload weights as new
    DF.model.load_state_dict(torch.load(new_weight_path))
    new_features = DF.get_features()
    inter_dist_after, _ = grad_confusion_pair(DF.model, new_features, [pair_ind1], [pair_ind2])
    print('After distance: ', inter_dist_after)


