import numpy as np
import torch
import torchvision

import loss
import utils
import dataset
from loss import *
from networks import Feat_resnet50_max_n, Full_Model
from evaluation.pumap import prepare_data, get_wrong_indices
import torch.nn as nn
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchsummary import summary
from torchcam.methods import SmoothGradCAMpp, GradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from PIL import Image
import scipy
from torch.autograd import Variable
import pickle
from scipy.stats import t
from utils import predict_batchwise
from influence_function import calc_confusion
from typing import Optional, List, Union, Tuple, Any
from influential_sample import InfluentialSample
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class GradCAMCustomize(GradCAMpp):

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:

        super().__init__(model,
            target_layer,
            input_shape,
            **kwargs)

    def _get_weights(self, scores, eigenvector) -> List[torch.Tensor]:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self._backprop(scores, eigenvector)
        # Alpha coefficient for each pixel
        grad_2 = [grad.pow(2) for grad in self.hook_g]
        grad_3 = [g2 * grad for g2, grad in zip(grad_2, self.hook_g)]
        # Watch out for NaNs produced by underflow
        spatial_dims = self.hook_a[0].ndim - 2  # type: ignore[union-attr]
        denom = [
            2 * g2 + (g3 * act).flatten(2).sum(-1)[(...,) + (None,) * spatial_dims]
            for g2, g3, act in zip(grad_2, grad_3, self.hook_a)
        ]
        nan_mask = [g2 > 0 for g2 in grad_2]
        alpha = grad_2
        for idx, d, mask in zip(range(len(grad_2)), denom, nan_mask):
            alpha[idx][mask].div_(d[mask])

        # Apply pixel coefficient in each weight
        return [
            a.squeeze_(0).mul_(torch.relu(grad.squeeze(0))).flatten(1).sum(-1)
            for a, grad in zip(alpha, self.hook_g)
        ]

    def _backprop(self, scores: torch.Tensor, eigenvec: torch.Tensor) -> None:
        # Backpropagate to get the gradients on the hooked layer
        eigenvec = eigenvec.to(scores.device)
        eigenvec.requires_grad = False
        loss = torch.dot(scores.squeeze(0), eigenvec)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def compute_cams(self, scores: torch.Tensor, eigenvec: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:

        # Get map weight & unsqueeze it
        weights = self._get_weights(scores, eigenvec)
        cams: List[torch.Tensor] = []

        for weight, activation in zip(weights, self.hook_a):
            missing_dims = activation.ndim - weight.ndim - 1  # type: ignore[union-attr]
            weight = weight[(...,) + (None,) * missing_dims]

            # Perform the weighted combination to get the CAM
            cam = torch.nansum(weight * activation.squeeze(0), dim=0)  # type: ignore[union-attr]
            # print("weights", weights)
            # print("activation", activation)

            if self._relu:
                cam = F.relu(cam, inplace=True)

            # Normalize the CAM
            if normalized:
                cam = self._normalize(cam)

            cams.append(cam)

        return cams

    def __call__(self, scores: torch.Tensor, eigenvec: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:
        return self.compute_cams(scores, eigenvec, normalized)

class XGradCAMCustomize(GradCAMCustomize):

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:

        super().__init__(model,
            target_layer,
            input_shape,
            **kwargs)

    def _get_weights(self, scores, eigenvector) -> List[torch.Tensor]:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self._backprop(scores, eigenvector)
        return [
            (grad * act).squeeze(0).flatten(1).sum(-1) / act.squeeze(0).flatten(1).sum(-1)
            for act, grad in zip(self.hook_a, self.hook_g)
        ]

def saliency(input, model, eigenvec):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    # forward pass to calculate predictions
    scores = model(input)
    eigenvec = eigenvec.to(scores.device)
    eigenvec.requires_grad = False
    loss = torch.dot(scores.squeeze(0), eigenvec)
    model.zero_grad()
    loss.backward(retain_graph=True)

    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())
    return slc

class DistinguishFeat(InfluentialSample):
    def __init__(self, dataset_name, seed, loss_type, config_name, sz_embedding=512):
        super().__init__(dataset_name, seed, loss_type, config_name,
                         'confusion', test_resize=False, sz_embedding=sz_embedding)
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
        v, phi = torch.eig(S, eigenvectors=True)
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
        E_X = torch.sum(feat_cls1, dim=0)
        E_Y = torch.sum(feat_cls2, dim=0)
        v, phi = torch.eig(E_XY - torch.outer(E_X, E_Y), eigenvectors=True)
        first_eigenveector = phi[:, 0].float()
        return first_eigenveector

    def CAM(self, interested_cls, eigenvector, dl, base_dir='XCAM'):
        assert len(interested_cls) == 2
        cam_extractor = XGradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
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
                img = to_pil_image(read_image(self.dl_ev.dataset.im_paths[indices[0]]))
                result = overlay_mask(img, to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)

                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                # plt.show()
                plt.savefig('./{}/{}/Confusion_{}_{}/{}/{}'.format(base_dir, self.dataset_name,
                                                                 interested_cls[0], interested_cls[1],
                                                                 y.item(),
                                                                 os.path.basename(self.dl_ev.dataset.im_paths[indices[0]])))

    def CAM_sample(self, interested_cls, helpful_ind, harmful_ind,
                   eigenvector, dl, base_dir='XCAM'):

        cam_extractor = XGradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
        os.makedirs('./{}/{}'.format(base_dir, self.dataset_name), exist_ok=True)

        for ct, (x, y, indices) in tqdm(enumerate(dl)):
            if indices.item() in helpful_ind:
                os.makedirs('./{}/{}/Confusion_{}_{}/{}'.format(base_dir, self.dataset_name, interested_cls[0], interested_cls[1], 'helpful'), exist_ok=True)
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
                os.makedirs('./{}/{}/Confusion_{}_{}/{}'.format(base_dir, self.dataset_name, interested_cls[0], interested_cls[1], 'harmful'), exist_ok=True)
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

    # def SaliencyMap_confuse(self, interested_cls, eigenvector):
    #     assert len(interested_cls) == 2
    #     for ct, (x, y, indices) in tqdm(enumerate(self.dl_ev)):
    #         os.makedirs('./SMap_confuse/{}'.format(self.dataset_name), exist_ok=True)
    #         os.makedirs('./SMap_confuse/{}/Confusion_{}_{}'.format(self.dataset_name, interested_cls[0], interested_cls[1]), exist_ok=True)
    #         if y.item() in interested_cls:
    #             os.makedirs('./SMap_confuse/{}/Confusion_{}_{}/{}'.format(self.dataset_name, interested_cls[0], interested_cls[1], y.item()), exist_ok=True)
    #
    #             x, y = x.cuda(), y.cuda()
    #             # Retrieve the CAM by passing the class index and the model output
    #             s_map = saliency(x, self.model, eigenvector)
    #
    #             # Resize the CAM and overlay it
    #             img = to_pil_image(read_image(self.dl_ev.dataset.im_paths[indices[0]]))
    #             result = overlay_mask(img, to_pil_image(s_map.detach().cpu(), mode='F'), alpha=0.35)
    #
    #             # Display it
    #             plt.imshow(result); plt.axis('off'); plt.tight_layout()
    #             # plt.show()
    #             plt.savefig('./SMap_confuse/{}/Confusion_{}_{}/{}/{}'.format(self.dataset_name,
    #                                                              interested_cls[0], interested_cls[1],
    #                                                              y.item(),
    #                                                              os.path.basename(self.dl_ev.dataset.im_paths[indices[0]])))
    #
    # def SaliencyMap(self, interested_cls, eigenvector):
    #     assert len(interested_cls) == 2
    #     for ct, (x, y, indices) in tqdm(enumerate(self.dl_ev)):
    #         os.makedirs('./SMap/{}'.format(self.dataset_name), exist_ok=True)
    #         os.makedirs('./SMap/{}/Confusion_{}_{}'.format(self.dataset_name, interested_cls[0], interested_cls[1]), exist_ok=True)
    #         if y.item() in interested_cls:
    #             os.makedirs('./SMap/{}/Confusion_{}_{}/{}'.format(self.dataset_name, interested_cls[0], interested_cls[1], y.item()), exist_ok=True)
    #
    #             x, y = x.cuda(), y.cuda()
    #             # Retrieve the CAM by passing the class index and the model output
    #             s_map = saliency(x, self.model, eigenvector)
    #
    #             # Resize the CAM and overlay it
    #             img = to_pil_image(read_image(self.dl_ev.dataset.im_paths[indices[0]]))
    #             result = overlay_mask(img, to_pil_image(s_map.detach().cpu(), mode='F'), alpha=0.35)
    #
    #             # Display it
    #             plt.imshow(result); plt.axis('off'); plt.tight_layout()
    #             # plt.show()
    #             plt.savefig('./SMap/{}/Confusion_{}_{}/{}/{}'.format(self.dataset_name,
    #                                                              interested_cls[0], interested_cls[1],
    #                                                              y.item(),
    #                                                              os.path.basename(self.dl_ev.dataset.im_paths[indices[0]])))

if __name__ == '__main__':
    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    i = 143; j = 140

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, sz_embedding)
    # feat_cls1 = DF.testing_embedding[DF.testing_label == i]
    # feat_cls2 = DF.testing_embedding[DF.testing_label == j]
    # confusion = calc_confusion(feat_cls1, feat_cls2, sqrt=True)  # get t instead of t^2
    # print(confusion.item())

    # eigenvec = DF.get_distinguish_feat(i, j)
    eigenvec = DF.get_confusing_feat(i, j)
    print(eigenvec.shape)
    '''Training'''
    # harmful = np.load('Influential_data/{}_{}_harmful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # helpful = np.load('Influential_data/{}_{}_helpful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '0'))
    # DF.CAM_sample(interested_cls=[i, j], helpful_ind=helpful, harmful_ind=harmful,
    #               eigenvector=eigenvec, dl=DF.dl_tr)

    '''Two testing classes'''
    DF.CAM(interested_cls=[i, j], eigenvector=eigenvec, dl=DF.dl_ev)
#