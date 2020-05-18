import torch
import torch.nn as nn
from collections import OrderedDict


class EyeNet(torch.nn.Module):

    def __init__(
        self,
        D_in,
        H,
        D_out,
        ):
        """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """

        super(EyeNet, self).__init__()
        self.model = nn.Sequential(OrderedDict({
            'layerin': nn.Linear(D_in, H),
            'relu': nn.ReLU(),
            'layerout': nn.Linear(H, D_out),
            'sigmoid': nn.Sigmoid(),
            }))

    def forward(self, x):
        """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary (differentiable) operations on Tensors.
    """

        y_pred = self.model(x)
        return y_pred
