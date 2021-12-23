from typing import Optional, List, Union, Tuple, Any
from torchcam.methods import GradCAM
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

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
        self.model.eval()

    def get_weights(self, scores: torch.Tensor, eigenvector: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self.backprop(scores, eigenvector)
        # Global average pool the gradients over spatial dimensions
        return [grad.squeeze(0).flatten(1).mean(-1) for grad in self.hook_g]

    def backprop(self, scores: torch.Tensor, eigenvec: torch.Tensor) -> None:
        # Backpropagate to get the gradients on the hooked layer
        eigenvec = eigenvec.to(scores.device)
        loss = (scores @ eigenvec.detach()).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        pass

    def compute_cams(self, scores: torch.Tensor, eigenvec: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:

        # Get map weight & unsqueeze it
        weights = self.get_weights(scores, eigenvec)
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

