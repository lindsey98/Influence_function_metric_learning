from typing import Optional, List, Union, Tuple, Any
from torchcam.methods import GradCAM
import torch
import torch.nn as nn
import torch.nn.functional as F

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



