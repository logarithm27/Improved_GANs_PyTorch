import torch
from torch import device
from torch.nn import *
import torch.nn.init as weight_initialization
from utilities import weight_normalization

DEVICE = device('cuda')


class Generator(Module):
    def __init__(self, noise_dimension, output_dimension=28 * 28):
        super(Generator, self).__init__()
        self.noise_dimension = noise_dimension
        self.dense_layer_1 = Sequential(Linear(noise_dimension, 500, bias=False),
                                        BatchNorm1d(500, affine=False,eps=1e-6, momentum=0.5),
                                        Softplus())
        self.dense_layer_2 = Sequential(Linear(500,500,bias=False),
                                        BatchNorm1d(500, affine=False, eps=1e-6, momentum=0.5),
                                        Softplus())
        self.weight_normalization_ = Sequential(weight_normalization(500,output_dimension),Softplus())
        weight_initialization.kaiming_uniform_(self.dense_layer_1[0].weight)
        weight_initialization.kaiming_uniform_(self.dense_layer_2[0].weight)

    def forward(self,batch_size):
        x = torch.rand(batch_size, self.noise_dimension).to(device=DEVICE)
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        x = self.weight_normalization_(x)
        return x

