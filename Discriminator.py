import torch
from torch import device
from torch import nn
from torch.nn import *
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu_
from utilities import weight_normalization
DEVICE = device('cuda')


class Discriminator(nn.Module):
    def __init__(self,input_dimension = 28*28, output_dimension = 10):
        super(Discriminator,self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.layers = ModuleList(
            [
                weight_normalization(input_dimension,1000),
                weight_normalization(1000, 500),
                weight_normalization(500, 250),
                weight_normalization(250, 250)
            ]
        )
        self.output_layer = weight_normalization(250, output_dimension)

    def forward(self, x, feature = False):
        x = x.view(-1, self.input_dimension)
        x = x.to(device=DEVICE)
        noise = torch.randn(x.size()) * 0.3 if self.training else torch.Tensor([0])
        noise = noise.to(device=DEVICE)
        x = x + noise
        for dense_layer in self.layers:
            x_feature = relu(dense_layer(x))
            x_feature = x_feature.to(device=DEVICE)
            noise = torch.randn(x_feature.size()) * 0.5 if self.training else torch.Tensor([0])
            noise = noise.to(device=DEVICE)
            x = x_feature + noise
        if feature:
            return x_feature, self.output_layer(x)
        return self.output_layer(x)

