import numpy as np
import torch
import torchvision
from matplotlib import cm

from loss import *
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image
from PIL import Image
from influential_sample import InfluentialSample
from CAM_methods import *
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
    def __init__(self, dataset_name, seed, loss_type, config_name, sz_embedding=512):
        super().__init__(dataset_name, seed, loss_type, config_name,
                         'confusion', test_resize=False, sz_embedding=sz_embedding)

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

    def CAM(self, interested_cls, eigenvector, dl, base_dir='CAM'):
        assert len(interested_cls) == 2
        cam_extractor = GradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
        for ct, (x, y, indices) in tqdm(enumerate(dl)):
            os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)
            os.makedirs('./{}/{}/Confusion_{}_{}'.format(base_dir, self.dataset_name, interested_cls[0], interested_cls[1]), exist_ok=True)
            if y.item() in interested_cls:
                os.makedirs('./{}/{}/Confusion_{}_{}/{}'.format(base_dir, self.dataset_name, interested_cls[0], interested_cls[1], y.item()), exist_ok=True)
                x, y = x.cuda(), y.cuda()
                out = self.model(x) # (1, sz_embed)
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out, eigenvector)
                # Resize the CAM and overlay it
                img = to_pil_image(read_image(dl.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                # plt.show()
                plt.savefig('./{}/{}/Confusion_{}_{}/{}/{}'.format(base_dir, self.dataset_name,
                                                                  interested_cls[0], interested_cls[1],
                                                                  y.item(),
                                                                  os.path.basename(dl.dataset.im_paths[indices[0]])))

    def CAM_sample(self, interested_cls, helpful_ind, harmful_ind,
                   eigenvector, dl, base_dir='CAM'):

        cam_extractor = GradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)

        for ct, (x, y, indices) in tqdm(enumerate(dl)):
            if indices.item() in helpful_ind:
                os.makedirs('./{}/{}/Confusion_{}_{}/{}'.format(base_dir, self.dataset_name,
                                                                interested_cls[0], interested_cls[1],
                                                                'helpful'), exist_ok=True)
                x, y = x.cuda(), y.cuda()
                out = self.model(x) # (1, sz_embed)
                activation_map = cam_extractor(out, eigenvector)

                img = to_pil_image(read_image(dl.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                # plt.show()
                plt.savefig('./{}/{}/Confusion_{}_{}/{}/{}'.format(base_dir, self.dataset_name,
                                                                  interested_cls[0], interested_cls[1],
                                                                  'helpful',
                                                                  os.path.basename(dl.dataset.im_paths[indices[0]])))
            elif indices.item() in harmful_ind:
                os.makedirs('./{}/{}/Confusion_{}_{}/{}'.format(base_dir, self.dataset_name,
                                                                interested_cls[0], interested_cls[1],
                                                                'harmful'), exist_ok=True)
                x, y = x.cuda(), y.cuda()
                out = self.model(x) # (1, sz_embed)
                activation_map = cam_extractor(out, eigenvector)
                img = to_pil_image(read_image(dl.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                # plt.show()
                plt.savefig('./{}/{}/Confusion_{}_{}/{}/{}'.format(base_dir, self.dataset_name,
                                                                 interested_cls[0], interested_cls[1],
                                                                 'harmful',
                                                                 os.path.basename(dl.dataset.im_paths[indices[0]])))


if __name__ == '__main__':
    # dataset_name = 'cub'
    # loss_type = 'ProxyNCA_pfix'
    # config_name = 'cub'
    # sz_embedding = 512
    # seed = 4
    # i = 117; j = 129

    # DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, sz_embedding)

    '''Analyze confusing features'''
    # eigenvec = DF.get_confusing_feat(i, j)
    # print(eigenvec.shape)

    '''Training'''
    # helpful = np.load('Influential_data/{}_{}_helpful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # harmful = np.load('Influential_data/{}_{}_harmful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # DF.CAM_sample(interested_cls=[i, j],
    #               helpful_ind=helpful, harmful_ind=harmful,
    #               eigenvector=eigenvec, dl=DF.dl_tr, base_dir='CAM+')

    '''Two testing classes'''
    # DF.CAM(interested_cls=[i, j], eigenvector=eigenvec, dl=DF.dl_ev, base_dir='CAM+')

    '''Analyze distingushing features'''
    dataset_name = 'cub+117_129'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    i = 117; j = 129

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, sz_embedding)
    eigenvec = DF.get_distinguish_feat(i, j)
    DF.CAM(interested_cls=[i, j], eigenvector=eigenvec,
           dl=DF.dl_ev, base_dir='CAM+_distinguish')
