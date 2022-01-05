
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from PIL import Image
from Influence_function.influential_sample import InfluentialSample
from Explaination.CAM_methods import *
from Influence_function.influence_function import *
from Explaination.background_removal import remove_background
import utils
import dataset
from torchvision import transforms
from dataset.utils import RGBAToRGB, ScaleIntensities
from utils import overlay_mask
from utils import predict_batchwise_debug
from evaluation import assign_by_euclidian_at_k_indices
import sklearn
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class DistinguishFeat(InfluentialSample):
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 test_crop=False, sz_embedding=512, epoch=40):

        super().__init__(dataset_name, seed, loss_type, config_name,
                          test_crop, sz_embedding, epoch)
        # FIXME: For analysis purpose, I disable centercrop data augmentation
        dataset_config = utils.load_config('dataset/config.json')
        self.data_transforms = transforms.Compose([
                    RGBAToRGB(),
                    transforms.Resize(dataset_config['transform_parameters']["sz_crop"]),
                    transforms.ToTensor(),
                    ScaleIntensities(*dataset_config['transform_parameters']["intensity_scale"]),
                    transforms.Normalize(
                        mean=dataset_config['transform_parameters']["mean"],
                        std=dataset_config['transform_parameters']["std"],
                    )
        ])

    def temporal_influence_func(self, wrong_indices, confuse_indices):

        '''Step 1: All confusion gradient to parameters'''
        theta_orig = self.model.module[-1].weight.data
        torch.cuda.empty_cache()
        all_features = self.get_features()  # (N, 2048)
        # Revise back the weights
        self.model.module[-1].weight.data = theta_orig
        theta = theta_orig.clone()

        # Record original inter-class distance
        inter_dist_orig, _ = grad_confusion_pair(self.model, all_features, wrong_indices, confuse_indices)  # dD/dtheta
        print("Original confusion: ", inter_dist_orig)

        # Optimization
        for _ in range(50):
            inter_dist, v = grad_confusion_pair(self.model, all_features, wrong_indices, confuse_indices)  # dD/dtheta
            print("Confusion: ", inter_dist)
            if inter_dist - inter_dist_orig >= 50.:  # FIXME: stopping criteria threshold selection
                break
            theta_new = theta + 4e-5 * v[0].to(theta.device)  # gradient ascent
            theta = theta_new
            self.model.module[-1].weight.data = theta

        self.model.module[-1].weight.data = theta_orig

        '''Step 2: Training class loss changes'''
        l_prev, l_cur = loss_change_train(self.model, self.criterion, self.dl_tr, theta_orig, theta)
        grad_loss = {'l_prev': l_prev, 'l_cur': l_cur}

        '''Step 3: Calc influence functions'''
        self.viz_2sample(self.dl_ev, wrong_indices[0], confuse_indices[0])
        influence_values = calc_influential_func_sample(grad_loss)
        influence_values = np.asarray(influence_values)
        training_sample_by_influence = influence_values.argsort()  # ascending
        self.viz_sample(training_sample_by_influence[:10])  # helpful
        self.viz_sample(training_sample_by_influence[-10:])  # harmful
        return training_sample_by_influence


    def GradAnalysis(self, wrong_cls, confuse_cls,
                     wrong_indices, confuse_indices,
                     dl, base_dir='Grad_Test'): # Only for confusion analysis

        assert len(wrong_indices) == len(confuse_indices)

        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
        os.makedirs('./{}/{}/{}_{}/'.format(base_dir, self.dataset_name,  wrong_cls, confuse_cls), exist_ok=True)
        model_copy = self._load_model()
        model_copy.eval()

        for ind1, ind2 in zip(wrong_indices, confuse_indices):
            cam_extractor1 = GradCAMCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer
            cam_extractor2 = GradCAMCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer

            # Get the two embeddings first
            cam_extractor1._hooks_enabled = True
            model_copy.zero_grad()
            emb1 = model_copy(dl.dataset.__getitem__(ind1)[0].unsqueeze(0).cuda())
            emb2 = model_copy(dl.dataset.__getitem__(ind2)[0].unsqueeze(0).cuda())
            activation_map1 = cam_extractor1(torch.dot(emb1.squeeze(0), emb2.detach().squeeze(0)))
            img1 = to_pil_image(read_image(dl.dataset.im_paths[ind1]))
            result1, _ = overlay_mask(img1, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)

            cam_extractor2._hooks_enabled = True
            model_copy.zero_grad()
            emb1 = model_copy(dl.dataset.__getitem__(ind1)[0].unsqueeze(0).cuda())
            emb2 = model_copy(dl.dataset.__getitem__(ind2)[0].unsqueeze(0).cuda())
            activation_map2 = cam_extractor2(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
            img2 = to_pil_image(read_image(dl.dataset.im_paths[ind2]))
            result2, _ = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)

            # Display it
            fig = plt.figure()
            fig.subplots_adjust(top=0.8)

            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(img1)
            plt.axis('off')

            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(img2)
            plt.axis('off')

            ax=fig.add_subplot(2, 2, 3)
            ax.imshow(result1)
            plt.axis('off')

            ax=fig.add_subplot(2, 2, 4)
            ax.imshow(result2)
            plt.axis('off')
            plt.show()
            # plt.savefig('./{}/{}/{}_{}/{}_{}.png'.format(base_dir, self.dataset_name,
            #                                              wrong_cls, confuse_cls,
            #                                              ind1, ind2))
            plt.close()


    # def GradAnalysis4Train(self,
    #                        helpful_indices, harmful_indices,
    #                        dl, base_dir='Confuse_Vis_Train'):
    #
    #     model_copy = self._load_model()
    #
    #     for ind in helpful_indices:
    #         cam_extractor = GradCAMCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer
    #
    #         # Get the two embeddings first
    #         cam_extractor._hooks_enabled = True
    #         model_copy.zero_grad()
    #
    #         cls_label = self.testing_label[ind]
    #         same_cls_indices = (self.testing_label == cls_label).to(torch.float32).nonzero().flatten()
    #         same_cls_emb = self.testing_embedding[same_cls_indices]
    #
    #         distances = sklearn.metrics.pairwise.pairwise_distances(same_cls_emb.detach().cpu().numpy())
    #         nn_ind = np.argsort(distances, axis=1)[:, 1][same_cls_indices == ind]
    #         nn_ind = same_cls_indices[nn_ind].item()
    #
    #         img = to_pil_image(read_image(dl.dataset.im_paths[ind]))
    #         img_nn = to_pil_image(read_image(dl.dataset.im_paths[nn_ind]))
    #
    #         out = model_copy(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         score = self.criterion.forward_score(out, torch.tensor([dl.dataset.__getitem__(ind)[1]]).cuda())
    #         activation_map = cam_extractor(score)
    #         result, _ = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)
    #
    #         fig = plt.figure()
    #         fig.subplots_adjust(top=0.8)
    #         ax = fig.add_subplot(2, 2, 1)
    #         ax.imshow(img)
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 2, 2)
    #         ax.imshow(img_nn)
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 2, 3)
    #         ax.imshow(result)
    #         plt.axis('off')
    #
    #         plt.suptitle('Class = {}, Ind = {}, NN Ind = {}'.format(cls_label, ind, nn_ind))
    #         # plt.show()
    #         plt.savefig('./{}/helpful_cls{}_{}.png'.format(base_dir, cls_label, ind))
    #         plt.close()

    def VisTrainNN(self, wrong_ind, confusion_ind,
                   interest_index, orig_NN_indices, orig_same_cls_NN_indices,
                   curr_NN_indices, curr_same_cls_NN_indices,
                   model_orig, model_curr,
                   dl,
                   base_dir):

        plt_dir = './{}/{}/{}_{}'.format(base_dir, self.dataset_name, wrong_ind, confusion_ind)
        os.makedirs(plt_dir, exist_ok=True)
        model_orig.eval(); model_curr.eval()
        # plotting layout
        num_rows = 2
        num_cols = len(orig_NN_indices) + 1

        fig = plt.figure()
        fig.subplots_adjust(top=0.8)

        ax = fig.add_subplot(num_rows, num_cols, 1)
        ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[interest_index])))
        ax.title.set_text('Ind = {}, Class = {}'.format(interest_index, dl.dataset.ys[interest_index]))
        plt.axis('off')

        for kk in range(len(orig_NN_indices)):
            ax = fig.add_subplot(num_rows, num_cols, kk+2)
            ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]])))

            emb1 = model_orig(dl.dataset.__getitem__(interest_index)[0].unsqueeze(0).cuda())
            emb2 = model_orig(dl.dataset.__getitem__(orig_NN_indices[kk])[0].unsqueeze(0).cuda())
            d = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()

            emb1_a = model_curr(dl.dataset.__getitem__(interest_index)[0].unsqueeze(0).cuda())
            emb2_a = model_curr(dl.dataset.__getitem__(orig_NN_indices[kk])[0].unsqueeze(0).cuda())
            d_a = (emb1_a.squeeze(0) - emb2_a.squeeze(0)).square().sum()

            ax.title.set_size(7)
            ax.title.set_text('Class = {}, D = {:.2f}=>{:.2f}'.format(dl.dataset.ys[orig_NN_indices[kk]], d, d_a))
            plt.axis('off')

        for kk in range(len(orig_same_cls_NN_indices)):
            ax = fig.add_subplot(num_rows, num_cols, kk+3+len(orig_NN_indices))
            ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_same_cls_NN_indices[kk]])))
            emb1 = model_orig(dl.dataset.__getitem__(interest_index)[0].unsqueeze(0).cuda())
            emb2 = model_orig(dl.dataset.__getitem__(orig_same_cls_NN_indices[kk])[0].unsqueeze(0).cuda())
            d = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()

            emb1_a = model_curr(dl.dataset.__getitem__(interest_index)[0].unsqueeze(0).cuda())
            emb2_a = model_curr(dl.dataset.__getitem__(orig_same_cls_NN_indices[kk])[0].unsqueeze(0).cuda())
            d_a = (emb1_a.squeeze(0) - emb2_a.squeeze(0)).square().sum()

            ax.title.set_size(7)
            ax.title.set_text('Class = {}, D = {:.2f}=>{:.2f}'.format(dl.dataset.ys[orig_same_cls_NN_indices[kk]], d, d_a))
            plt.axis('off')

        plt.show()
        # plt.savefig('./{}/{}.png'.format(plt_dir, interest_index))
        plt.close()

    def VisTrain(self, wrong_ind, confusion_ind,
                 interest_indices, orig_NN_indices, curr_NN_indices,
                 model1, model2,
                 dl,
                 base_dir):

        plt_dir = './{}/{}/{}_{}'.format(base_dir, self.dataset_name, wrong_ind, confusion_ind)
        os.makedirs(plt_dir, exist_ok=True)
        model1.eval(); model2.eval()

        for kk, ind in enumerate(interest_indices):
            fig = plt.figure()
            fig.subplots_adjust(top=0.8)

            ax = fig.add_subplot(3, 3, 1)
            ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[ind])))
            ax.title.set_text('Ind = {}, Class = {}'.format(ind, dl.dataset.ys[ind]))
            plt.axis('off')

            ax = fig.add_subplot(3, 3, 2)
            ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]])))
            ax.title.set_text('Ind = {}, Class = {}'.format(orig_NN_indices[kk], dl.dataset.ys[orig_NN_indices[kk]]))
            plt.axis('off')

            ax = fig.add_subplot(3, 3, 3)
            ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]])))
            ax.title.set_text('Ind = {}, Class = {}'.format(curr_NN_indices[kk], dl.dataset.ys[curr_NN_indices[kk]]))
            plt.axis('off')

            cam_extractor1 = GradCAMCustomize(model1,
                                              target_layer=model1.module[0].base.layer4)  # to last layer
            cam_extractor2 = GradCAMCustomize(model2,
                                              target_layer=model2.module[0].base.layer4)  # to last layer
            cam_extractor1._hooks_enabled = True
            model1.zero_grad()
            emb1 = model1(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
            emb2 = model1(dl.dataset.__getitem__(orig_NN_indices[kk])[0].unsqueeze(0).cuda())
            activation_map1 = cam_extractor1(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
            img1 = to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]]))
            result1, _ = overlay_mask(img1, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)
            d1 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()

            cam_extractor2._hooks_enabled = True
            model2.zero_grad()
            emb1 = model2(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
            emb2 = model2(dl.dataset.__getitem__(curr_NN_indices[kk])[0].unsqueeze(0).cuda())
            activation_map2 = cam_extractor2(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
            img2 = to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]]))
            result2, _ = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)
            d2 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()

            ax = fig.add_subplot(3, 3, 5)
            ax.imshow(result1)
            ax.title.set_text('Distance = {:.4f}'.format(d1))
            plt.axis('off')

            ax = fig.add_subplot(3, 3, 6)
            ax.imshow(result2)
            ax.title.set_text('Distance = {:.4f}'.format(d2))
            plt.axis('off')

            # Affinity to proxy
            proxy = self.criterion.proxies[dl.dataset.ys[ind]].detach()
            cam_extractor1 = GradCAMCustomize(model1,
                                              target_layer=model1.module[0].base.layer4)  # to last layer
            cam_extractor2 = GradCAMCustomize(model2,
                                              target_layer=model2.module[0].base.layer4)  # to last layer
            cam_extractor1._hooks_enabled = True
            model1.zero_grad()
            emb1 = model1(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
            activation_map1 = cam_extractor1(torch.dot(emb1.squeeze(0), proxy.to(emb1.device)))
            img = to_pil_image(read_image(dl.dataset.im_paths[ind]))
            result1, _ = overlay_mask(img, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)

            cam_extractor2._hooks_enabled = True
            model2.zero_grad()
            emb1 = model2(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
            activation_map2 = cam_extractor2(torch.dot(emb1.squeeze(0), proxy.to(emb1.device)))
            result2, _ = overlay_mask(img, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)

            ax = fig.add_subplot(3, 3, 8)
            ax.imshow(result1)
            plt.axis('off')

            ax = fig.add_subplot(3, 3, 9)
            ax.imshow(result2)
            plt.axis('off')
            # plt.show()
            plt.savefig('./{}/{}.png'.format(plt_dir, ind))
            plt.close()

    # def HelpfulTrain(self,
    #                  wrong_ind, confusion_ind,
    #                  helpful_indices, orig_NN_indices, curr_NN_indices,
    #                  cur_weight_path,
    #                  dl,
    #                  base_dir):
    #
    #     plt_dir = './{}/{}/{}_{}'.format(base_dir, self.dataset_name, wrong_ind, confusion_ind)
    #     os.makedirs(plt_dir, exist_ok=True)
    #     model_copy = self._load_model()
    #     model_copy.load_state_dict(torch.load(cur_weight_path))
    #
    #     model_copy_orig = self._load_model()
    #     model_copy.eval()
    #     model_copy_orig.eval()
    #
    #     for kk, ind in enumerate(helpful_indices):
    #         fig = plt.figure()
    #         fig.subplots_adjust(top=0.8)
    #
    #         ax = fig.add_subplot(2, 3, 1)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[ind])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(ind, dl.dataset.ys[ind]))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 3, 2)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(orig_NN_indices[kk], dl.dataset.ys[orig_NN_indices[kk]]))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 3, 3)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(curr_NN_indices[kk], dl.dataset.ys[curr_NN_indices[kk]]))
    #         plt.axis('off')
    #
    #         cam_extractor1 = GradCAMCustomize(model_copy_orig,
    #                                           target_layer=model_copy_orig.module[0].base.layer4)  # to last layer
    #         cam_extractor2 = GradCAMCustomize(model_copy,
    #                                           target_layer=model_copy.module[0].base.layer4)  # to last layer
    #         cam_extractor1._hooks_enabled = True
    #         model_copy_orig.zero_grad()
    #         emb1 = model_copy_orig(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         emb2 = model_copy_orig(dl.dataset.__getitem__(orig_NN_indices[kk])[0].unsqueeze(0).cuda())
    #         activation_map1 = cam_extractor1(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
    #         img1 = to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]]))
    #         result1, _ = overlay_mask(img1, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)
    #         d1 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()
    #
    #         cam_extractor2._hooks_enabled = True
    #         model_copy.zero_grad()
    #         emb1 = model_copy(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         emb2 = model_copy(dl.dataset.__getitem__(curr_NN_indices[kk])[0].unsqueeze(0).cuda())
    #         activation_map2 = cam_extractor2(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
    #         img2 = to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]]))
    #         result2, _ = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)
    #         d2 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()
    #
    #         ax = fig.add_subplot(2, 3, 5)
    #         ax.imshow(result1)
    #         ax.title.set_text('Distance = {:.4f}'.format(d1))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 3, 6)
    #         ax.imshow(result2)
    #         ax.title.set_text('Distance = {:.4f}'.format(d2))
    #         plt.axis('off')
    #         plt.savefig('./{}/{}.png'.format(plt_dir, ind))
    #         plt.close()
    #
    # def SacrificedTrain(self, wrong_ind, confusion_ind,
    #                     sacrifice_indices, orig_NN_indices, curr_NN_indices,
    #                     cur_weight_path,
    #                     dl,
    #                     base_dir):
    #
    #     plt_dir = './{}/{}/{}_{}'.format(base_dir, self.dataset_name, wrong_ind, confusion_ind)
    #     os.makedirs(plt_dir, exist_ok=True)
    #     model_copy = self._load_model()
    #     model_copy.load_state_dict(torch.load(cur_weight_path))
    #     model_copy_orig = self._load_model()
    #     model_copy.eval()
    #     model_copy_orig.eval()
    #
    #     for kk, ind in enumerate(sacrifice_indices):
    #         fig = plt.figure()
    #         fig.subplots_adjust(top=0.8)
    #
    #         ax = fig.add_subplot(2, 3, 1)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[ind])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(ind, dl.dataset.ys[ind]))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 3, 2)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(orig_NN_indices[kk], dl.dataset.ys[orig_NN_indices[kk]]))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 3, 3)
    #         ax.imshow(to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]])))
    #         ax.title.set_text('Ind = {}, Class = {}'.format(curr_NN_indices[kk], dl.dataset.ys[curr_NN_indices[kk]]))
    #         plt.axis('off')
    #
    #         cam_extractor1 = GradCAMCustomize(model_copy_orig,
    #                                           target_layer=model_copy_orig.module[0].base.layer4)  # to last layer
    #         cam_extractor2 = GradCAMCustomize(model_copy,
    #                                           target_layer=model_copy.module[0].base.layer4)  # to last layer
    #         cam_extractor1._hooks_enabled = True
    #         model_copy_orig.zero_grad()
    #         emb1 = model_copy_orig(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         emb2 = model_copy_orig(dl.dataset.__getitem__(orig_NN_indices[kk])[0].unsqueeze(0).cuda())
    #         activation_map1 = cam_extractor1(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
    #         img1 = to_pil_image(read_image(dl.dataset.im_paths[orig_NN_indices[kk]]))
    #         result1, _ = overlay_mask(img1, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)
    #         d1 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()
    #
    #         cam_extractor2._hooks_enabled = True
    #         model_copy.zero_grad()
    #         emb1 = model_copy(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
    #         emb2 = model_copy(dl.dataset.__getitem__(curr_NN_indices[kk])[0].unsqueeze(0).cuda())
    #         activation_map2 = cam_extractor2(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
    #         img2 = to_pil_image(read_image(dl.dataset.im_paths[curr_NN_indices[kk]]))
    #         result2, _ = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)
    #         d2 = (emb1.squeeze(0) - emb2.squeeze(0)).square().sum()
    #
    #         ax = fig.add_subplot(2, 3, 5)
    #         ax.imshow(result1)
    #         ax.title.set_text('Distance = {:.4f}'.format(d1))
    #         plt.axis('off')
    #
    #         ax = fig.add_subplot(2, 3, 6)
    #         ax.imshow(result2)
    #         ax.title.set_text('Distance = {:.4f}'.format(d2))
    #         plt.axis('off')
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
    '''Step 1: Visualize all pairs, Find interesting pairs'''
    # confusion_class_pairs = DF.get_confusion_class_pairs()
    # for cls_idx in range(len(confusion_class_pairs)):
    #     for pair_idx in [0]:
    #         wrong_cls = confusion_class_pairs[cls_idx][pair_idx][0]
    #         confusion_cls = confusion_class_pairs[cls_idx][pair_idx][1]
    #         print(wrong_cls, confusion_cls)
    #         pred = DF.testing_nn_label.flatten(); label = DF.testing_label.flatten()
    #         nn_indices = DF.testing_nn_indices.flatten()
    #
    #         wrong_as_confusion_cls_indices = np.where((pred == confusion_cls) & (label == wrong_cls))[0]
    #         wrong_indices = wrong_as_confusion_cls_indices
    #         confuse_indices = nn_indices[wrong_as_confusion_cls_indices]
    #
    #         DF.GradAnalysis( wrong_cls, confusion_cls,
    #                          wrong_indices, confuse_indices,
    #                          DF.dl_ev, base_dir='Confuse_Vis')
    # exit()

    '''Step 2: Human Confuse, Model Confuse: Do Influential sample training'''
    wrong_index = 2541; confuse_index = 2661
    base_dir = 'Confuse_pair_influential_data'
    new_weight_path = 'models/dvi_data_{}_{}_loss{}_{}_{}/ResNet_512_Model/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(
                                                                                                            dataset_name,
                                                                                                           seed,
                                                                                                           'ProxyNCA_pfix_confusion_{}_{}'.format(
                                                                                                           wrong_index, confuse_index),
                                                                                                           10, -10,
                                                                                                           1, dataset_name,
                                                                                                           dataset_name,
                                                                                                           512, seed) # reload weights as new

    # training_sample_by_influence = DF.temporal_influence_func([wrong_index], [confuse_index])
    # helpful_indices = training_sample_by_influence[:10]
    # harmful_indices = training_sample_by_influence[-10:]
    # os.makedirs(base_dir, exist_ok=True)
    # np.save('./{}/helpful_indices_{}_{}'.format(base_dir, wrong_index, confuse_index), helpful_indices)
    # np.save('./{}/harmful_indices_{}_{}'.format(base_dir, wrong_index, confuse_index), harmful_indices)
    # exit()

    '''Step 3: Train the model'''
    # Run in shell

    '''Step 4: Sanity check: Whether the confusion pairs are pulled far apart, Whether the confusion samples is pulled closer to correct neighbor'''
    # DF.model = DF._load_model()  # reload the original weights
    # new_features = DF.get_features()
    # inter_dist_orig, _ = grad_confusion_pair(DF.model, new_features, [wrong_index], [confuse_index])
    # print("Original distance: ", inter_dist_orig)
    #
    # distances = sklearn.metrics.pairwise.pairwise_distances(DF.testing_embedding)  # (N_test, N_test)
    # diff_cls_mask = (DF.testing_label[:, None] != DF.testing_label).detach().cpu().numpy().nonzero()
    # distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
    # nn_indices_same_cls = np.argsort(distances, axis=1)[:, 1]
    # wrong_nn_indices_same_cls = nn_indices_same_cls[wrong_index]
    # d1, _ = grad_confusion_pair(DF.model, new_features, [wrong_index], [wrong_nn_indices_same_cls])
    # print("Before distance to correct neighbor: ", d1)
    #
    # DF.model.load_state_dict(torch.load(new_weight_path))
    # new_features = DF.get_features()
    # inter_dist_after, _ = grad_confusion_pair(DF.model, new_features, [wrong_index], [confuse_index])
    # print("After distance: ", inter_dist_after)
    #
    # new_test_embedding, _, _ = predict_batchwise_debug(DF.model, DF.dl_ev)
    # distances = sklearn.metrics.pairwise.pairwise_distances(new_test_embedding)  # (N_test, N_test)
    # distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
    # nn_indices_same_cls = np.argsort(distances, axis=1)[:, 1]
    # wrong_nn_indices_same_cls = nn_indices_same_cls[wrong_index]
    # d1, _ = grad_confusion_pair(DF.model, new_features, [wrong_index], [wrong_nn_indices_same_cls])
    # print("After distance to correct neighbor: ", d1)

    '''Step 5: Find cases (normal training samples not in harmful/helpful) where its original 1st NN is no longer its 1st NN'''
    helpful_indices = np.load('./{}/helpful_indices_{}_{}.npy'.format(base_dir, wrong_index, confuse_index))
    harmful_indices = np.load('./{}/harmful_indices_{}_{}.npy'.format(base_dir, wrong_index, confuse_index))
    # predict training 1st NN (original)
    train_nn_indices_orig, train_nn_label_orig = assign_by_euclidian_at_k_indices(DF.train_embedding, DF.train_label, 1)
    # predict training 1st NN within the same class (original)
    distances = sklearn.metrics.pairwise.pairwise_distances(DF.train_embedding)  # (N_train, N_train)
    diff_cls_mask = (DF.train_label[:, None] != DF.train_label).detach().cpu().numpy().nonzero()
    distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
    train_nn_indices_same_cls_orig = np.argsort(distances, axis=1)[:, 1]

    new_model = DF._load_model()
    new_model.load_state_dict(torch.load(new_weight_path))
    train_embedding_curr, train_label, _ = predict_batchwise_debug(new_model, DF.dl_tr)
    # predict training 1st NN (after training)
    train_nn_indices_curr, train_nn_label_curr = assign_by_euclidian_at_k_indices(train_embedding_curr, train_label, 1)
    # predict training 1st NN within the same class(after training)
    distances = sklearn.metrics.pairwise.pairwise_distances(train_embedding_curr)  # (N_test, N_test)
    distances[diff_cls_mask[0], diff_cls_mask[1]] = distances.max() + 1
    train_nn_indices_same_cls_curr = np.argsort(distances, axis=1)[:, 1]

    # Find whether 1st NN has changed from correct to wrong class (Similar -> Dissimilar)
    inconsistent_wrong_indices = ((train_nn_label_orig.flatten() == train_label.detach().cpu().numpy()) &
                                  (train_nn_label_orig.flatten() != train_nn_label_curr.flatten())).nonzero()[0]
    inconsistent_wrong_indices = set(inconsistent_wrong_indices.tolist()) - set(helpful_indices.tolist()) - set(harmful_indices.tolist())
    inconsistent_wrong_indices = list(inconsistent_wrong_indices)
    print(len(inconsistent_wrong_indices))

    # Plot out (Sacrificed samples, its original NN, and its current NN)
    model_orig = DF._load_model()
    model_curr = DF._load_model()
    model_curr.load_state_dict(torch.load(new_weight_path))

    DF.VisTrain(wrong_ind=wrong_index, confusion_ind=confuse_index,
                       interest_indices=inconsistent_wrong_indices,
                       orig_NN_indices=train_nn_indices_orig.flatten()[inconsistent_wrong_indices],
                       curr_NN_indices=train_nn_indices_curr.flatten()[inconsistent_wrong_indices],
                       model1=model_orig, model2=model_curr,
                       dl=DF.dl_tr, base_dir='Confuse_sacrifice_train')

    # Plot out helpful training
    DF.VisTrain(wrong_ind=wrong_index, confusion_ind=confuse_index,
                interest_indices=helpful_indices,
                orig_NN_indices=train_nn_indices_same_cls_orig.flatten()[helpful_indices],
                curr_NN_indices=train_nn_indices_same_cls_curr.flatten()[helpful_indices],
                model1=model_orig, model2=model_curr,
                dl=DF.dl_tr,
                base_dir='Confuse_helpful_train'
            )

    DF.VisTrain(wrong_ind=wrong_index, confusion_ind=confuse_index,
                    interest_indices=harmful_indices,
                    orig_NN_indices=train_nn_indices_same_cls_orig.flatten()[harmful_indices],
                    curr_NN_indices=train_nn_indices_same_cls_curr.flatten()[harmful_indices],
                    model1=model_orig, model2=model_curr,
                    dl=DF.dl_tr,
                    base_dir='Confuse_harmful_train'
                )