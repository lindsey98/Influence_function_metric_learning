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
from influence_function import inverse_hessian_product, grad_confusion, grad_alltrain, calc_confusion, calc_influential_func, calc_intravar, grad_intravar
import pickle
from scipy.stats import t
from utils import predict_batchwise
from typing import Optional, List, Union, Tuple, Any
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

            if self._relu:
                cam = F.relu(cam, inplace=True)

            # Normalize the CAM
            if normalized:
                cam = self._normalize(cam)

            cams.append(cam)

        return cams

    def __call__(self, scores: torch.Tensor, eigenvec: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:
        return self.compute_cams(scores, eigenvec, normalized)


class DistinguishFeat():
    def __init__(self, dataset_name, seed, loss_type, config_name, sz_embedding=512):

        self.folder = 'dvi_data_{}_{}_loss{}/'.format(dataset_name, seed, loss_type)
        self.model_dir = '{}/ResNet_{}_Model'.format(self.folder, sz_embedding)

        # load data
        self.dl_tr, self.dl_ev = prepare_data(data_name=dataset_name, config_name=config_name, root=self.folder, save=False, batch_size=1)
        self.dataset_name = dataset_name
        self.seed = seed
        self.loss_type = loss_type
        self.sz_embedding = sz_embedding
        self.criterion = self._load_criterion()
        self.model = self._load_model()
        self.train_embedding, self.train_label, self.testing_embedding, self.testing_label = self._load_data()
        pass

    def _load_model(self):
        feat = Feat_resnet50_max_n()
        emb = torch.nn.Linear(2048, self.sz_embedding)
        model = torch.nn.Sequential(feat, emb)
        model = nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(torch.load('{}/Epoch_{}/{}_{}_trainval_{}_{}.pth'.format(self.model_dir, 40, self.dataset_name, self.dataset_name, self.sz_embedding, self.seed)))
        return model

    def _load_criterion(self):
        proxies = torch.load('{}/Epoch_{}/proxy.pth'.format(self.model_dir, 40), map_location='cpu')['proxies'].detach()
        if 'ProxyNCA_prob_orig' in self.loss_type:
            criterion = loss.ProxyNCA_prob_orig(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding, scale=3)
        elif 'ProxyNCA_pfix' in self.loss_type:
            criterion = loss.ProxyNCA_pfix(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding, scale=3)
        elif 'ProxyAnchor' in self.loss_type:
            criterion = loss.Proxy_Anchor(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding)
        elif 'ProxyAnchor_pfix' in self.loss_type:
            criterion = loss.ProxyAnchor_pfix(nb_classes=self.dl_tr.dataset.nb_classes(), sz_embed=self.sz_embedding)
        else:
            raise NotImplementedError
        criterion.proxies.data = proxies
        criterion.cuda()
        return criterion
    def cache_embedding(self):
        embedding, label, indices = predict_batchwise(self.model, self.dl_tr)
        torch.save(embedding, '{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, 40))
        torch.save(label, '{}/Epoch_{}/training_labels.pth'.format(self.model_dir, 40))

        testing_embedding, testing_label, testing_indices = predict_batchwise(self.model, self.dl_ev)
        torch.save(testing_embedding, '{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, 40))
        torch.save(testing_label, '{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, 40))
    def _load_data(self):
        try:
            train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, 40))  # high dimensional embedding
        except FileNotFoundError:
            self.cache_embedding()
        train_embedding = torch.load('{}/Epoch_{}/training_embeddings.pth'.format(self.model_dir, 40))  # high dimensional embedding
        train_label = torch.load('{}/Epoch_{}/training_labels.pth'.format(self.model_dir, 40))
        testing_embedding = torch.load('{}/Epoch_{}/testing_embeddings.pth'.format(self.model_dir, 40))  # high dimensional embedding
        testing_label = torch.load('{}/Epoch_{}/testing_labels.pth'.format(self.model_dir, 40))

        return train_embedding, train_label, testing_embedding, testing_label

    @torch.no_grad()
    def get_distinguish_feat(self, cls1, cls2):

        feat_cls1 = self.testing_embedding[self.testing_label == cls1] # (N, sz_embedding)
        feat_cls2 = self.testing_embedding[self.testing_label == cls2] # (N, sz_embedding)
        # centering
        feat_cls1_centered = feat_cls1 - feat_cls1.mean(0).unsqueeze(0)
        feat_cls2_centered = feat_cls2 - feat_cls2.mean(0).unsqueeze(0)
        S = torch.zeros((feat_cls1.size()[-1], feat_cls1.size()[-1])) # (sz_embed, sz_embed)
        for i in range(len(feat_cls1_centered)):
            for j in range(len(feat_cls2_centered)):
                outer_prod = torch.outer(feat_cls1_centered[i]-feat_cls2_centered[j], feat_cls1_centered[i]-feat_cls2_centered[j])
                S += outer_prod
        v, phi = torch.eig(S, eigenvectors=True)
        first_eigenveector = phi[:, 0].float()
        return first_eigenveector

    def CAM(self, interested_cls, eigenvector):
        assert len(interested_cls) == 2
        cam_extractor = GradCAMCustomize(self.model, target_layer=self.model.module[0].base.layer1)
        for ct, (x, y, indices) in tqdm(enumerate(self.dl_ev)):
            os.makedirs('./CAM', exist_ok=True)
            os.makedirs('./CAM/Confusion_{}_{}'.format(interested_cls[0], interested_cls[1]), exist_ok=True)
            if y.item() in interested_cls:
                os.makedirs('./CAM/Confusion_{}_{}/{}'.format(interested_cls[0], interested_cls[1], y.item()), exist_ok=True)

                x, y = x.cuda(), y.cuda()
                out = self.model(x) # (1, sz_embed)
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out, eigenvector)
                # Resize the CAM and overlay it
                img = read_image(self.dl_ev.dataset.im_paths[indices[0]])
                result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].detach().cpu(), mode='F'), alpha=0.5)
                # Display it
                plt.imshow(result); plt.axis('off'); plt.tight_layout()
                plt.show()
                # plt.savefig('./CAM/Confusion_{}_{}/{}/{}'.format(interested_cls[0], interested_cls[1],
                #                                                  y.item(),
                #                                                  os.path.basename(self.dl_ev.dataset.im_paths[indices[0]]))
                #             )


if __name__ == '__main__':
    dataset_name = 'cub+172_178'
    loss_type = 'ProxyNCA_prob_orig'
    config_name = 'cub'
    sz_embedding = 512
    seed = 0

    DF = DistinguishFeat(dataset_name, seed, loss_type, config_name, sz_embedding)
    eigenvec = DF.get_distinguish_feat(172, 178)
    DF.CAM([172, 178], eigenvec)
#