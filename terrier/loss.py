import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma:float=2.0,
        weights:Tensor|None=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.weights = weights

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        probabilities = F.softmax(predictions, dim=-1)
        loss = -(1-probabilities)** self.gamma * torch.log(probabilities)

        # Weights
        if self.weights is not None:
            self.weights = self.weights.to(target.device)
            loss *= torch.gather(self.weights, -1, target)

        return loss.mean()