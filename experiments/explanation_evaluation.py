
from Influence_function.influence_function import MCScalableIF
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
import numpy as np
from explanation.CAM_methods import *
from utils import overlay_mask
import matplotlib.transforms as mtrans
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class Explanation4EIF(MCScalableIF):
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 test_crop=False, sz_embedding=512, epoch=40):

        super().__init__(dataset_name, seed, loss_type, config_name,
                          test_crop, sz_embedding, epoch)

    def vis_harmful_training(self, wrong_index, confuse_index, base_dir='Grad_Test'):

        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
        model_copy = self._load_model()
        model_copy.eval()

        # Get the two embeddings first
        img1 = to_pil_image(read_image(self.dl_ev.dataset.im_paths[wrong_index]))
        img2 = to_pil_image(read_image(self.dl_ev.dataset.im_paths[confuse_index]))
        # Display it
        fig = plt.figure()
        plt.rcParams.update({'axes.titlesize': 'small', 'font.weight': 'bold'})
        upper_bound = []
        lower_bound = []

        ax = fig.add_subplot(3, 5, 1)
        ax.imshow(img1)
        plt.tight_layout()
        ax.title.set_text('Test Ind = {} \n Class = {}'.format(wrong_index, self.dl_ev.dataset.ys[wrong_index]))
        upper_bound.append(ax.bbox.ymax / fig.bbox.ymax)
        lower_bound.append(ax.bbox.ymin / fig.bbox.ymax)
        plt.axis('off')

        ax = fig.add_subplot(3, 5, 2)
        ax.imshow(img2)
        plt.tight_layout()
        ax.title.set_text('Test Ind = {} \n Class = {}'.format(confuse_index, self.dl_ev.dataset.ys[confuse_index]))
        upper_bound.append(ax.bbox.ymax / fig.bbox.ymax)
        lower_bound.append(ax.bbox.ymin / fig.bbox.ymax)
        x0 = ax.bbox.xmax / fig.bbox.xmax
        plt.axis('off')

        # load harmful training
        if not os.path.exists('./{}/{}_influence_values_{}_{}.npy'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index)) or \
                not os.path.exists('./{}/{}_helpful_indices_{}_{}.npy'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index)) or \
                not './{}/{}_harmful_indices_{}_{}.npy'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index):

            mean_deltaL_deltaD = self.MC_estimate_forpair([wrong_index, confuse_index], num_thetas=1, steps=50)
            influence_values = np.asarray(mean_deltaL_deltaD)

            harmful_indices = np.where(influence_values > 0)[0]
            helpful_indices = np.where(influence_values < 0)[0]
            np.save('./{}/{}_harmful_indices_{}_{}'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index),
                    harmful_indices)
            np.save('./{}/{}_helpful_indices_{}_{}'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name),
                                                           loss_type, wrong_index, confuse_index),
                    helpful_indices)
            np.save('./{}/{}_influence_values_{}_{}'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index),
                    influence_values)
        else:
            harmful_indices = np.load('./{}/{}_harmful_indices_{}_{}.npy'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index))
            helpful_indices = np.load('./{}/{}_helpful_indices_{}_{}.npy'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index))
            influence_values = np.load('./{}/{}_influence_values_{}_{}.npy'.format('Confuse_pair_influential_data/{}'.format(self.dataset_name), loss_type, wrong_index, confuse_index))

        influence4harmful = influence_values[harmful_indices]
        influence4helpful = influence_values[helpful_indices]
        influence4harmful_sort = influence4harmful.argsort()[::-1] # sort in descending
        top_harmful_indices = harmful_indices[influence4harmful_sort[:5]] # get topk harmful training indices
        influence4helpful_sort = influence4helpful.argsort() # sort in ascending
        top_helpful_indices = helpful_indices[influence4helpful_sort[:5]] # get topk harmful training indices

        top_harmful_labels = self.train_label[top_harmful_indices] # get their labels
        top_harmful_proxies = self.criterion.proxies[top_harmful_labels.long(), :] # get their learnt proxies
        top_helpful_labels = self.train_label[top_helpful_indices] # get their labels
        top_helpful_proxies = self.criterion.proxies[top_helpful_labels.long(), :] # get their learnt proxies

        for it, harmful_ind in enumerate(top_harmful_indices):

            harmful_img = to_pil_image(read_image(self.dl_tr.dataset.im_paths[harmful_ind]))
            ax=fig.add_subplot(3, 5, 6+it)
            ax.imshow(harmful_img)
            plt.tight_layout()
            ax.title.set_text('Class = {}'.format(harmful_ind, self.dl_tr.dataset.ys[harmful_ind]))
            upper_bound.append(ax.bbox.ymax / fig.bbox.ymax)
            lower_bound.append(ax.bbox.ymin/ fig.bbox.ymax)
            plt.axis('off')

        for it, harmful_ind in enumerate(top_harmful_indices):

            # cam_extractor = GradCAMCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer
            cam_extractor = GuidedGradCAMCustomize(model_copy, target_layer=model_copy.module[0].base.layer4)  # to last layer
            cam_extractor._hooks_enabled = True
            model_copy.zero_grad()
            inputs = self.dl_tr.dataset.__getitem__(harmful_ind)[0].unsqueeze(0).cuda()
            inputs.requires_grad = True
            emb = model_copy(inputs)
            proxy = top_harmful_proxies[it].cuda()
            # activation_map = cam_extractor(torch.dot(emb.squeeze(0), proxy.detach()))
            activation_map = cam_extractor(inputs=inputs,
                                           scores=torch.dot(emb.squeeze(0), proxy.detach()))
            harmful_img = to_pil_image(read_image(self.dl_tr.dataset.im_paths[harmful_ind]))
            # result_img, _ = overlay_mask(harmful_img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)
            # result_img, _ = overlay_mask(harmful_img, to_pil_image(activation_map[0][0].detach().cpu(), mode='F'), alpha=0.5)
            result_img = activation_map[0][0].detach().cpu().permute(1,2,0)

            ax=fig.add_subplot(3, 5, 11+it)
            ax.imshow(result_img)
            plt.tight_layout()
            upper_bound.append(ax.bbox.ymax / fig.bbox.ymax)
            lower_bound.append(ax.bbox.ymin/ fig.bbox.ymax)
            plt.axis('off')
            del cam_extractor

        plt.tight_layout()
        # plt.savefig('./{}/{}/{}_{}_GradCAM.pdf'.format(base_dir, self.dataset_name, wrong_index, confuse_index))
        # plt.close()
        plt.show()
        print('haha')


if __name__ == '__main__':
    loss_type = 'ProxyNCA_prob_orig'; sz_embedding = 512; epoch = 40; test_crop = False
    dataset_name = 'cub'; config_name = 'cub'; seed = 0
    # dataset_name = 'cars'; config_name = 'cars'; seed = 3
    # dataset_name = 'inshop'; config_name = 'inshop'; seed = 4

    IS = Explanation4EIF(dataset_name, seed, loss_type, config_name, test_crop)

    result_log_file = 'Confuse_pair_influential_data/{}_pairs.txt'.format(IS.dataset_name)
    for line in open(result_log_file).readlines():
        wrong_index, confuse_index = int(line.split('\t')[0]), int(line.split('\t')[1])
        IS.vis_harmful_training(wrong_index, confuse_index, base_dir='Grad_Test')
        break
