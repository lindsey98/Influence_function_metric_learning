from typing import Optional, List, Union, Tuple, Any
from torchcam.methods import GradCAM
import torch
import torch.nn as nn
import torch.nn.functional as F

from captum._utils.common import _format_input, _format_output, _is_tuple
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from explanation.guided_gradient import GuidedBackprop
from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.log import log_usage
import warnings

class GradCustomize(GradCAM):

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

    def get_weights(self, scores: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self.backprop(scores)
        return [grad.squeeze(0) for grad in self.hook_g]

    def backprop(self, scores: torch.Tensor) -> None:
        # Backpropagate to get the gradients on the hooked layer
        loss = scores
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def compute_gradients(self, scores: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:

        # Get map weight & unsqueeze it
        weights = self.get_weights(scores)
        cams: List[torch.Tensor] = []

        for weight, activation in zip(weights, self.hook_a):

            cam = torch.nansum(weight, dim=0) # combine over channel dimension

            if self._relu:
                cam = F.relu(cam, inplace=True)

            # Normalize
            if normalized:
                cam = self._normalize(cam)

            cams.append(cam)

        return cams

    def __call__(self, scores: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:
        return self.compute_gradients(scores, normalized)



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

    def get_weights(self, scores: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self.backprop(scores)
        return [grad.squeeze(0).flatten(1).mean(-1) for grad in self.hook_g]

    def backprop(self, scores: torch.Tensor) -> None:
        # Backpropagate to get the gradients on the hooked layer
        loss = scores
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def compute_gradients(self, scores: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:

        # Get map weight & unsqueeze it
        weights = self.get_weights(scores)
        cams: List[torch.Tensor] = []

        for weight, activation in zip(weights, self.hook_a):
            missing_dims = activation.ndim - weight.ndim - 1  # type: ignore[union-attr]
            weight = weight[(...,) + (None,) * missing_dims]

            # Perform the weighted combination to get the CAM
            cam = torch.nansum(weight * activation.squeeze(0), dim=0)  # type: ignore[union-attr]

            if self._relu:
                cam = F.relu(cam, inplace=True)

            # Normalize
            if normalized:
                cam = self._normalize(cam)

            cams.append(cam)

        return cams

    def __call__(self, scores: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:
        return self.compute_gradients(scores, normalized)


class GuidedGradCAMCustomize(GradCAM):

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
        self.guided_backprop = GuidedBackprop(model)

    def get_weights(self, scores: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self.backprop(scores)
        return [grad.squeeze(0).flatten(1).mean(-1) for grad in self.hook_g]

    def backprop(self, scores: torch.Tensor) -> None:
        # Backpropagate to get the gradients on the hooked layer
        loss = scores
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        pass

    def compute_gradients(self, scores: torch.Tensor, normalized: bool = True) -> List[torch.Tensor]:

        # Get map weight & unsqueeze it
        weights = self.get_weights(scores)
        cams: List[torch.Tensor] = []

        for weight, activation in zip(weights, self.hook_a):
            missing_dims = activation.ndim - weight.ndim - 1  # type: ignore[union-attr]
            weight = weight[(...,) + (None,) * missing_dims]

            # Perform the weighted combination to get the CAM
            cam = torch.nansum(weight * activation.squeeze(0), dim=0)  # type: ignore[union-attr]

            if self._relu:
                cam = F.relu(cam, inplace=True)

            # Normalize
            if normalized:
                cam = self._normalize(cam)

            cams.append(cam)

        return cams

    def __call__(self, inputs: torch.Tensor, scores: torch.Tensor,
                 normalized: bool = True) -> List[torch.Tensor]:
        gradcam = self.compute_gradients(scores, normalized)
        gradcam = gradcam[0]
        guided_backprop_attr = inputs.grad

        output_attr: List[torch.Tensor] = []
        for i in range(len(inputs)):
            try:
                output_attr.append(
                    guided_backprop_attr[i]
                    * LayerAttribution.interpolate(gradcam.unsqueeze(0).unsqueeze(0), inputs.shape[2:], interpolate_mode="bicubic")
                )
                pass
            except Exception:
                warnings.warn(
                    "Couldn't appropriately interpolate GradCAM attributions for some "
                    "input tensors, returning empty tensor for corresponding "
                    "attributions."
                )
                output_attr.append(torch.empty(0))

        return output_attr
