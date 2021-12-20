import numpy as np
import torch
import torchvision

from loss import *
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchcam.methods import GradCAMpp, GradCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from PIL import Image
from typing import Optional, List, Union, Tuple, Any
from influential_sample import InfluentialSample
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class GradCAMCustomize(GradCAM):

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

    def _get_weights(self, scores: torch.Tensor, eigenvector: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self._backprop(scores, eigenvector)
        # Global average pool the gradients over spatial dimensions
        return [grad.squeeze(0).flatten(1).mean(-1) for grad in self.hook_g]

    def _backprop(self, scores: torch.Tensor, eigenvec: torch.Tensor) -> None:
        # Backpropagate to get the gradients on the hooked layer
        eigenvec = eigenvec.to(scores.device)
        eigenvec.requires_grad = False
        loss = torch.dot(scores.squeeze(0), eigenvec) # alpha^T embedding
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

            if self._relu:
                cam = F.relu(cam, inplace=True)

            # Normalize the CAM
            if normalized:
                cam = self._normalize(cam)

            cams.append(cam)

        return cams

    def __call__(self, scores: torch.Tensor, eigenvec: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:
        return self.compute_cams(scores, eigenvec, normalized)

class GradCAMppCustomize(GradCAMCustomize):

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

class LayerCAMCustomize(GradCAMCustomize):

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
        self._backprop(scores, eigenvector)
        return [torch.relu(grad).squeeze(0) for grad in self.hook_g]

    @staticmethod
    def _scale_cams(cams: List[torch.Tensor], gamma: float = 2.) -> List[torch.Tensor]:
        # cf. Equation 9 in the paper
        return [torch.tanh(gamma * cam) for cam in cams]


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
        E_X = torch.sum(feat_cls1, dim=0); E_Y = torch.sum(feat_cls2, dim=0)
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
    dataset_name = 'cub'
    loss_type = 'ProxyNCA_pfix'
    config_name = 'cub'
    sz_embedding = 512
    seed = 4
    i = 143; j = 140

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, sz_embedding)

    '''Analyze confusing features'''
    eigenvec = DF.get_confusing_feat(i, j)
    # print(eigenvec.shape)

    '''Training'''
    # helpful = np.load('Influential_data/{}_{}_helpful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '3'))
    # harmful = np.load('Influential_data/{}_{}_harmful_testcls{}.npy'.format('cub', 'ProxyNCA_pfix', '3'))
    # DF.CAM_sample(interested_cls=[i, j],
    #               helpful_ind=helpful, harmful_ind=harmful,
    #               eigenvector=eigenvec, dl=DF.dl_tr)

    '''Two testing classes'''
    DF.CAM(interested_cls=[i, j], eigenvector=eigenvec, dl=DF.dl_ev, base_dir='CAM')

    '''Analyze distingushing features'''
    # dataset_name = 'cub+143_140'
    # loss_type = 'ProxyNCA_pfix'
    # config_name = 'cub'
    # sz_embedding = 512
    # seed = 4
    # i = 143; j = 140
    #
    # DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, sz_embedding)
    # eigenvec = DF.get_distinguish_feat(i, j)
    # DF.CAM(interested_cls=[i, j], eigenvector=eigenvec,
    #        dl=DF.dl_ev, base_dir='CAM_distinguish')
