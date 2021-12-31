
import os
import matplotlib.pyplot as plt
import numpy as np
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
from utils import predict_batchwise
from evaluation import assign_by_euclidian_at_k_indices
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

    def temporal_influence_func(self, wrong_cls, confuse_cls):

        '''Step 1: All confusion gradient to parameters'''
        theta_orig = self.model.module[-1].weight.data
        torch.cuda.empty_cache()
        all_features = self.get_features()  # (N, 2048)
        # Revise back the weights
        self.model.module[-1].weight.data = theta_orig
        theta = theta_orig.clone()

        # Record original inter-class distance
        inter_dist_orig, _ = grad_confusion(self.model, all_features, wrong_cls, confuse_cls,
                                            self.testing_nn_label, self.testing_label,
                                            self.testing_nn_indices)  # dD/dtheta
        print("Original confusion: ", inter_dist_orig)

        # Optimization
        for _ in range(50):
            inter_dist, v = grad_confusion(self.model, all_features, wrong_cls, confuse_cls,
                                           self.testing_nn_label, self.testing_label,
                                           self.testing_nn_indices)  # dD/dtheta
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
        influence_values = calc_influential_func_sample(grad_loss)
        influence_values = np.asarray(influence_values)
        training_sample_by_influence = influence_values.argsort()  # ascending
        return training_sample_by_influence

    def GradAnalysis(self, wrong_cls, confuse_cls,
                     wrong_indices, confuse_indices,
                     dl, base_dir='Grad_Test'): # Only for confusion analysis

        assert len(wrong_indices) == len(confuse_indices)

        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
        os.makedirs('./{}/{}/{}_{}/'.format(base_dir, self.dataset_name,  wrong_cls, confuse_cls), exist_ok=True)
        model_copy = self._load_model()

        for ind1, ind2 in zip(wrong_indices, confuse_indices):
            cam_extractor1 = GradCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer
            cam_extractor2 = GradCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer

            # Get the two embeddings first
            cam_extractor1._hooks_enabled = True
            model_copy.zero_grad()
            emb1 = model_copy(dl.dataset.__getitem__(ind1)[0].unsqueeze(0).cuda())
            emb2 = model_copy(dl.dataset.__getitem__(ind2)[0].unsqueeze(0).cuda())
            activation_map1 = cam_extractor1(torch.dot(emb1.squeeze(0), emb2.detach().squeeze(0)))
            img1 = to_pil_image(read_image(dl.dataset.im_paths[ind1]))
            result1 = overlay_mask(img1, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)

            cam_extractor2._hooks_enabled = True
            model_copy.zero_grad()
            emb1 = model_copy(dl.dataset.__getitem__(ind1)[0].unsqueeze(0).cuda())
            emb2 = model_copy(dl.dataset.__getitem__(ind2)[0].unsqueeze(0).cuda())
            activation_map2 = cam_extractor2(torch.dot(emb1.detach().squeeze(0), emb2.squeeze(0)))
            img2 = to_pil_image(read_image(dl.dataset.im_paths[ind2]))
            result2 = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)

            # Get euclidean distance
            before_distance = (emb1.detach() - emb2.detach()).square().sum().sqrt()

            # Display it
            fig = plt.figure()
            fig.subplots_adjust(top=0.8)
            ax=fig.add_subplot(1,2,1)
            ax.imshow(result1)
            plt.axis('off')

            ax=fig.add_subplot(1,2,2)
            ax.imshow(result2)
            plt.axis('off')

            plt.suptitle('D = {:.4f}'.format(before_distance), fontsize=10)
            # plt.show()
            plt.savefig('./{}/{}/{}_{}/{}_{}.png'.format(base_dir, self.dataset_name,
                                                         wrong_cls, confuse_cls,
                                                         ind1, ind2))
            plt.close()

        # # Remove background
        # img1_after = remove_background(dl.dataset.im_paths[ind1])
        # img2_after = remove_background(dl.dataset.im_paths[ind2])
        #
        # img1_after_trans = self.data_transforms(img1_after)
        # img2_after_trans = self.data_transforms(img2_after)
        #
        # emb1_after = self.model(img1_after_trans.unsqueeze(0).cuda()).detach()
        # emb2_after = self.model(img2_after_trans.unsqueeze(0).cuda()).detach()
        # after_distance = (emb1_after - emb2_after).square().sum().sqrt()
        #
        # ax=fig.add_subplot(2,2,3)
        # ax.imshow(img1_after)
        # plt.axis('off')
        #
        # ax=fig.add_subplot(2,2,4)
        # ax.imshow(img2_after)
        # plt.axis('off')
        # plt.suptitle('Before D = {:.4f}, After D = {:.4f}'.format(before_distance, after_distance), fontsize=10)
        # plt.savefig('./{}/{}/{}_{}/{}_{}.png'.format(base_dir, self.dataset_name,
        #                                                 interested_cls[0], interested_cls[1],
        #                                                 ind1, ind2))


    def GradAnalysis4Train(self, wrong_cls, confuse_cls,
                           dl, base_dir='Confuse_Vis_Train'):

        helpful_indices = np.load('./{}/{}/{}_{}/helpful_indices.npy'.format(base_dir, self.dataset_name, wrong_cls, confuse_cls))
        harmful_indices = np.load('./{}/{}/{}_{}/harmful_indices.npy'.format(base_dir, self.dataset_name, wrong_cls, confuse_cls))
        model_copy = self._load_model()

        for ind in helpful_indices:
            cam_extractor = GradCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer

            # Get the two embeddings first
            cam_extractor._hooks_enabled = True
            model_copy.zero_grad()
            out = model_copy(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
            cls_label = torch.tensor([dl.dataset.__getitem__(ind)[1]]).cuda()
            score = self.criterion.forward_score(out, cls_label)
            activation_map = cam_extractor(score)
            img = to_pil_image(read_image(dl.dataset.im_paths[ind]))
            result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

            fig = plt.figure()
            fig.subplots_adjust(top=0.8)
            plt.imshow(result)
            plt.axis('off')
            # plt.show()
            plt.savefig('./{}/{}/{}_{}/helpful_{}.png'.format(base_dir, self.dataset_name,
                                                         wrong_cls, confuse_cls,
                                                         ind))
            plt.close()

        for ind in harmful_indices:
            cam_extractor = GradCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer

            # Get the two embeddings first
            cam_extractor._hooks_enabled = True
            model_copy.zero_grad()
            out = model_copy(dl.dataset.__getitem__(ind)[0].unsqueeze(0).cuda())
            cls_label = torch.tensor([dl.dataset.__getitem__(ind)[1]]).cuda()
            score = self.criterion.forward_score(out, cls_label)
            activation_map = cam_extractor(score)
            img = to_pil_image(read_image(dl.dataset.im_paths[ind]))
            result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

            fig = plt.figure()
            fig.subplots_adjust(top=0.8)
            plt.imshow(result)
            plt.axis('off')
            # plt.show()
            plt.savefig('./{}/{}/{}_{}/harmful_{}.png'.format(base_dir, self.dataset_name,
                                                         wrong_cls, confuse_cls,
                                                         ind))
            plt.close()




if __name__ == '__main__':

    dataset_name = 'cars'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cars'
    sz_embedding = 512
    seed = 4
    test_crop = False

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, test_crop)

    '''Analyze confusing features'''
    confusion_class_pairs = DF.get_confusion_class_pairs()
    for cls_idx in range(len(confusion_class_pairs)):
        for pair_idx in range(len(confusion_class_pairs[cls_idx])):
            wrong_cls = confusion_class_pairs[cls_idx][pair_idx][0]
            confusion_cls = confusion_class_pairs[cls_idx][pair_idx][1]
            print(wrong_cls, confusion_cls)
            pred = DF.testing_nn_label.flatten(); label = DF.testing_label.flatten()
            nn_indices = DF.testing_nn_indices.flatten()

            wrong_as_confusion_cls_indices = np.where((pred == confusion_cls) & (label == wrong_cls))[0]
            wrong_indices = wrong_as_confusion_cls_indices
            confuse_indices = nn_indices[wrong_as_confusion_cls_indices]

            DF.GradAnalysis(
                         wrong_cls, confusion_cls,
                         wrong_indices, confuse_indices,
                         DF.dl_ev, base_dir='Confuse_Vis')

            base_dir = 'Confuse_Vis_Train'
            os.makedirs('./{}/{}'.format(base_dir, DF.dataset_name), exist_ok=True)
            os.makedirs('./{}/{}/{}_{}/'.format(base_dir, DF.dataset_name, wrong_cls, confusion_cls), exist_ok=True)
            training_sample_by_influence = DF.temporal_influence_func(wrong_cls, [confusion_cls])
            helpful_indices = training_sample_by_influence[:50]
            harmful_indices = training_sample_by_influence[-50:]
            np.save(
                './{}/{}/{}_{}/helpful_indices'.format(base_dir, DF.dataset_name, wrong_cls, confusion_cls),
                helpful_indices)
            np.save(
                './{}/{}/{}_{}/harmful_indices'.format(base_dir, DF.dataset_name, wrong_cls, confusion_cls),
                harmful_indices)
            DF.GradAnalysis4Train(wrong_cls, confusion_cls, DF.dl_tr)

