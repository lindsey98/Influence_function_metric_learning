from matplotlib import cm

import os
import matplotlib.pyplot as plt
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
from dataset.utils import RGBAToRGB
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.7) -> Image.Image:

    """Overlay a colormapped mask on a background image

    Example::
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> from torchcam.utils import overlay_mask
        >>> img = ...
        >>> cam = ...
        >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    # Overlay the image with the mask
    img = np.asarray(img)
    if len(img.shape) < 3: # create a dummy axis if img is single channel
        img = img[:, :, np.newaxis]
    overlayed_img = Image.fromarray((alpha * img + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


class DistinguishFeat(InfluentialSample):
    def __init__(self, dataset_name, seed, loss_type, config_name,
                 measure, test_crop=False, sz_embedding=512, epoch=40):

        super().__init__(dataset_name, seed, loss_type, config_name,
                         measure, test_crop, sz_embedding, epoch)
        # FIXME: For analysis purpose, I disable centercrop data augmentation
        dataset_config = utils.load_config('dataset/config.json')
        self.data_transforms = transforms.Compose([
                    RGBAToRGB(),
                    transforms.Resize(dataset_config['transform_parameters']["sz_crop"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
        ])
        pass

    def get_dist_between_classes(self, cls1, cls2):
        testing_embedding, testing_label, _ = utils.predict_batchwise(self.model, self.dl_ev)
        feat_cls1 = testing_embedding[testing_label == cls1]
        feat_cls2 = testing_embedding[testing_label == cls2]
        indices_cls1 = torch.arange(len(self.testing_embedding))[self.testing_label == cls1]
        indices_cls2 = torch.arange(len(self.testing_embedding))[self.testing_label == cls2]
        D = torch.cdist(feat_cls1, feat_cls2, p=2)

        # most confusing samples
        D_flatten = D.flatten()
        index_flatten = torch.tensor([[indices_cls1[i], indices_cls2[j]] for i in range(D.size()[0]) for j in range(D.size()[1])])

        sort_idx = torch.argsort(D_flatten)
        minimal_D = D_flatten[sort_idx]
        minimal_index = index_flatten[sort_idx]

        return minimal_D, minimal_index

    @torch.no_grad()
    def get_pc_feat(self, cls):
        feat_cls = self.testing_embedding[self.testing_label == cls]
        feat_mean = feat_cls.mean(0)
        S = torch.zeros((feat_cls.size()[-1], feat_cls.size()[-1])) # (sz_embed, sz_embed)
        for i in range(len(feat_cls)):
            S += torch.outer(feat_cls[i] - feat_mean, feat_cls[i] - feat_mean)
        v, phi = torch.eig(S / (feat_cls.size()[0] - 1), eigenvectors=True)
        first_eigenveector = phi[:, 0].float()
        return first_eigenveector

    @torch.no_grad()
    def get_distinguish_feat(self, cls1, cls2):

        feat_cls1 = self.testing_embedding[self.testing_label == cls1] # (N, sz_embedding)
        feat_cls2 = self.testing_embedding[self.testing_label == cls2] # (N, sz_embedding)
        # centering
        S = torch.zeros((feat_cls1.size()[-1], feat_cls1.size()[-1])) # (sz_embed, sz_embed)
        for i in range(len(feat_cls1)):
            for j in range(len(feat_cls2)):
                outer_prod = torch.outer(feat_cls1[i]-feat_cls2[j], feat_cls1[i]-feat_cls2[j])
                S += outer_prod
        v, phi = torch.eig(S / (feat_cls1.size()[0] * feat_cls2.size()[0]), eigenvectors=True)
        first_eigenveector = phi[:, 0].float()
        return first_eigenveector

    @torch.no_grad()
    def get_confusing_feat(self, cls1, cls2):

        feat_cls1 = self.testing_embedding[self.testing_label == cls1] # (N, sz_embedding)
        feat_cls2 = self.testing_embedding[self.testing_label == cls2] # (N, sz_embedding)
        # centering
        E_XY = torch.zeros((feat_cls1.size()[-1], feat_cls1.size()[-1])) # (sz_embed, sz_embed)
        for i in range(len(feat_cls1)):
            for j in range(len(feat_cls2)):
                outer_prod = torch.outer(feat_cls1[i], feat_cls2[j])
                E_XY += outer_prod
        E_XY = E_XY / (feat_cls1.size()[0] * feat_cls2.size()[0])
        E_X = torch.mean(feat_cls1, dim=0); E_Y = torch.mean(feat_cls2, dim=0)
        v, phi = torch.eig(E_XY - torch.outer(E_X, E_Y), eigenvectors=True)
        first_eigenveector = phi[:, 0].float()
        return first_eigenveector

    def dominant_samples(self, interested_cls, coeff, dl, base_dir='CAM'):
        torch.cuda.empty_cache()
        cam_extractor = GradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)

        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
        os.makedirs('./{}/{}/{}_{}_dominant_feat'.format(base_dir, self.dataset_name, self.measure, interested_cls),
                    exist_ok=True)

        feat_cls = self.testing_embedding[self.testing_label == interested_cls] # (N, sz_embed)
        proj = torch.matmul(feat_cls, coeff)
        sample_indices = torch.arange(len(self.testing_embedding))[self.testing_label == interested_cls]

        # rank and visualize
        sort_indices = sample_indices[torch.argsort(proj, descending=True)] # descending
        top_bottom_indices = torch.cat((sort_indices[:10], sort_indices[-10:]))

        result_vis = []
        orig_vis = []
        for ct, (x, y, indices) in tqdm(enumerate(dl)):
            if indices.item() in top_bottom_indices:
                x, y = x.cuda(), y.cuda()
                out = self.model(x)  # (1, sz_embed)
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out, coeff)
                # Resize the CAM and overlay it
                img = to_pil_image(read_image(dl.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)
                orig_vis.append(img)
                result_vis.append(result)
            if len(result_vis) == len(top_bottom_indices):
                break

        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(result_vis[i])
            plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig('./{}/{}/{}_{}_dominant_feat/{}.png'.format(base_dir, self.dataset_name, self.measure, interested_cls, 'cam'))

        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(orig_vis[i])
            plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig('./{}/{}/{}_{}_dominant_feat/{}.png'.format(base_dir, self.dataset_name, self.measure, interested_cls, 'original'))

    def CAM(self, interested_cls, coeff, dl, base_dir='CAM'):
        assert len(interested_cls) == 2
        cam_extractor = GradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
        for ct, (x, y, indices) in tqdm(enumerate(dl)):
            os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
            os.makedirs('./{}/{}/{}_{}_{}'.format(base_dir, self.dataset_name, self.measure, interested_cls[0], interested_cls[1]), exist_ok=True)
            if y.item() in interested_cls:
                os.makedirs('./{}/{}/{}_{}_{}/{}'.format(base_dir, self.dataset_name, self.measure, interested_cls[0], interested_cls[1], y.item()), exist_ok=True)
                x, y = x.cuda(), y.cuda()
                out = self.model(x) # (1, sz_embed)
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out, coeff)
                # Resize the CAM and overlay it
                img = to_pil_image(read_image(dl.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                # plt.show()
                plt.savefig('./{}/{}/{}_{}_{}/{}/{}'.format(base_dir, self.dataset_name,
                                                            self.measure,
                                                            interested_cls[0], interested_cls[1],
                                                            y.item(),
                                                            os.path.basename(dl.dataset.im_paths[indices[0]])))

    def CAM_helpful_harmful(self, interested_cls, helpful_ind, harmful_ind,
                            coeff, dl, base_dir='CAM'):

        cam_extractor = GradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)

        for ct, (x, y, indices) in tqdm(enumerate(dl)):
            if indices.item() in helpful_ind:
                if self.measure == 'confusion':
                    os.makedirs('./{}/{}/{}_{}_{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                                    interested_cls[0], interested_cls[1],
                                                                    'helpful'), exist_ok=True)
                else:
                    os.makedirs('./{}/{}/{}_{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                             interested_cls,
                                                             'helpful'), exist_ok=True)
                x, y = x.cuda(), y.cuda()
                out = self.model(x) # (1, sz_embed)
                activation_map = cam_extractor(out, coeff)

                img = to_pil_image(read_image(dl.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                # plt.show()
                if self.measure == 'confusion':
                    plt.savefig('./{}/{}/{}_{}_{}/{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                                      interested_cls[0], interested_cls[1],
                                                                      'helpful',
                                                                      os.path.basename(dl.dataset.im_paths[indices[0]])))
                else:
                    plt.savefig('./{}/{}/{}_{}/{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                                interested_cls,
                                                                'helpful',
                                                                os.path.basename(dl.dataset.im_paths[indices[0]])))
            elif indices.item() in harmful_ind:
                if self.measure == 'confusion':
                    os.makedirs('./{}/{}/{}_{}_{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                             interested_cls[0], interested_cls[1],
                                                             'harmful'), exist_ok=True)
                else:
                    os.makedirs('./{}/{}/{}_{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                          interested_cls,
                                                          'harmful'), exist_ok=True)
                x, y = x.cuda(), y.cuda()
                out = self.model(x) # (1, sz_embed)
                activation_map = cam_extractor(out, coeff)
                img = to_pil_image(read_image(dl.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                # plt.show()
                if self.measure == 'confusion':
                    plt.savefig('./{}/{}/{}_{}_{}/{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                                interested_cls[0], interested_cls[1],
                                                                'harmful',
                                                                os.path.basename(dl.dataset.im_paths[indices[0]])))
                else:
                    plt.savefig('./{}/{}/{}_{}/{}/{}'.format(base_dir, self.dataset_name, self.measure,
                                                             interested_cls,
                                                             'harmful',
                                                             os.path.basename(dl.dataset.im_paths[indices[0]])))

    def background_removal(self, interested_cls,
                           ind1, ind2, dl, base_dir='CAM_sample'): # Only for confusion analysis

        assert len(interested_cls) == 2
        cam_extractor = GradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
        os.makedirs('./{}/{}/{}_{}_{}/'.format(base_dir, self.dataset_name, self.measure, interested_cls[0], interested_cls[1]), exist_ok=True)

        # Get the two embeddings first
        for _, (x, y, indices) in tqdm(enumerate(self.dl_ev)):
            if indices.item() == ind1.item():
                emb1 = self.model(x.cuda())
            elif indices.item() == ind2.item():
                emb2 = self.model(x.cuda())
            else:
                if 'emb1' in locals() and 'emb2' in locals():
                    break

        emb1_detach = emb1.detach().squeeze(0)
        emb2_detach = emb2.detach().squeeze(0)
        before_distance = (emb1_detach - emb2_detach).square().sum().sqrt()
        activation_map1 = cam_extractor(emb1, emb2_detach)
        img1 = to_pil_image(read_image(dl.dataset.im_paths[ind1]))
        result1 = overlay_mask(img1, to_pil_image(activation_map1[0].detach().cpu(), mode='F'), alpha=0.5)

        activation_map2 = cam_extractor(emb2, emb1_detach)
        img2 = to_pil_image(read_image(dl.dataset.im_paths[ind2]))
        result2 = overlay_mask(img2, to_pil_image(activation_map2[0].detach().cpu(), mode='F'), alpha=0.5)

        # Display it
        fig = plt.figure()
        ax=fig.add_subplot(2,2,1)
        ax.imshow(result1)
        plt.axis('off'); plt.tight_layout()

        ax=fig.add_subplot(2,2,2)
        ax.imshow(result2)
        plt.axis('off'); plt.tight_layout()

        # Remove background
        img1_after = remove_background(dl.dataset.im_paths[ind1])
        img2_after = remove_background(dl.dataset.im_paths[ind2])

        img1_after_trans = self.data_transforms(img1_after)
        img2_after_trans = self.data_transforms(img2_after)
        print(img1_after_trans.shape)
        print(img2_after_trans.shape)

        emb1_after = self.model(img1_after_trans.unsqueeze(0).cuda()).detach().squeeze(0)
        emb2_after = self.model(img2_after_trans.unsqueeze(0).cuda()).detach().squeeze(0)
        after_distance = (emb1_after - emb2_after).square().sum().sqrt()

        ax=fig.add_subplot(2,2,3)
        ax.imshow(img1_after)
        plt.axis('off'); plt.tight_layout()

        ax=fig.add_subplot(2,2,4)
        ax.imshow(img2_after)
        plt.axis('off'); plt.tight_layout()
        plt.suptitle('Before D = {:.4f}, After D = {:.4f}'.format(before_distance, after_distance), fontsize=10)
        plt.show()
        # plt.savefig('./{}/{}/{}_{}_{}/{}_{}.png'.format(base_dir, self.dataset_name, self.measure,
        #                                                   interested_cls[0], interested_cls[1],
        #                                                   ind1,
        #                                                   ind2))


if __name__ == '__main__':
    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    measure = 'confusion'
    test_crop = False
    i = 143; j = 140
    # i = 160

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, measure, test_crop)

    '''###### For Confusion Analysis #####'''
    '''Analyze confusing features'''
    # eigenvec = DF.get_confusing_feat(i, j)
    # print(eigenvec.shape)

    '''Training'''
    # helpful = np.load('Influential_data/{}_{}_helpful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # harmful = np.load('Influential_data/{}_{}_harmful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # DF.CAM_sample(interested_cls=[i, j],
    #               helpful_ind=helpful, harmful_ind=harmful,
    #               eigenvector=eigenvec, dl=DF.dl_tr, base_dir='CAM')

    '''Two testing classes'''
    # DF.CAM(interested_cls=[i, j], eigenvector=eigenvec, dl=DF.dl_ev, base_dir='CAM')

    '''Analyze distingushing features'''
    # dataset_name = 'cub+117_129'
    # loss_type = 'ProxyNCA_pfix'
    # config_name = 'cub'
    # sz_embedding = 512
    # seed = 4
    # i = 117; j = 129
    #
    # DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, sz_embedding)
    # eigenvec = DF.get_distinguish_feat(i, j)
    # DF.CAM(interested_cls=[i, j], coeff=eigenvec,
    #        dl=DF.dl_ev, base_dir='CAM_distinguish')

    '''Analyze two confusing samples in specific'''
    minimal_D, minimal_idx_pair = DF.get_dist_between_classes(i, j)
    for pair_idx in range(10):
        ind1, ind2 = minimal_idx_pair[pair_idx]
        # os.makedirs(os.path.join('Background_erase', '{}_{}_{}_{}').format(dataset_name, DF.measure, i, j), exist_ok=True)
        # shutil.copyfile(DF.dl_ev.dataset.im_paths[ind1],
        #                 os.path.join('Background_erase', '{}_{}_{}_{}', '{}.png').format(dataset_name, DF.measure, i, j, ind1))
        # shutil.copyfile(DF.dl_ev.dataset.im_paths[ind2],
        #                 os.path.join('Background_erase', '{}_{}_{}_{}', '{}.png').format(dataset_name, DF.measure, i, j, ind2))

        DF.background_removal(interested_cls=(i, j), ind1=ind1, ind2=ind2, dl=DF.dl_ev)


    '''###### For Intra Class Variance Analysis #####'''
    '''Analyze confusing features'''
    # eigenvec = DF.get_pc_feat(i)
    # print(eigenvec.shape)

    '''Training'''
    # helpful = np.load('Influential_data/{}_{}_helpful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # harmful = np.load('Influential_data/{}_{}_harmful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # DF.CAM_helpful_harmful(interested_cls=i,
    #               helpful_ind=helpful, harmful_ind=harmful,
    #               eigenvector=eigenvec, dl=DF.dl_tr, base_dir='CAM')

    '''Analyze most significantly varying direction'''
    # DF.dominant_samples(i, eigenvec, DF.dl_ev)


    # '''###### Background removal experiment #####'''
    # from evaluation.pumap import load_single_sample
    # ind1, ind2 = 2568, 2404
    # x1 = load_single_sample(config_name, img_path='Background_erase/{}_confusion_{}_{}/{}_e.png'.format(dataset_name, i, j, ind1), test_resize=True)
    # x2 = load_single_sample(config_name, img_path='Background_erase/{}_confusion_{}_{}/{}_e.png'.format(dataset_name, i, j, ind2), test_resize=True)
    # with torch.no_grad():
    #     emb1 = DF.model(x1.unsqueeze(0).cuda()).detach().cpu()
    #     emb2 = DF.model(x2.unsqueeze(0).cuda()).detach().cpu()
    # print('Previous distance', (DF.testing_embedding[ind1] - DF.testing_embedding[ind2]).square().sum().sqrt().item())
    # print('After distance', (emb1 - emb2).square().sum().sqrt().item())