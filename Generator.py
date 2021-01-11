import torch
from torch import device
from torch.nn import *
import torch.nn.init as weight_initialization
from utilities import weight_normalization

# Make computations over CPU OR GPU
GPU = 'cuda'
CPU = 'cpu'


class Generator(Module):
    def __init__(self, noise_dimension, output_dimension=28 * 28):
        super(Generator, self).__init__()
        self.noise_dimension = noise_dimension
        # first hidden layer with noise dimension as input and 500 neurons in output
        self.dense_layer_1 = Sequential(Linear(noise_dimension, 500, bias=False),
                                        BatchNorm1d(500, affine=False,eps=1e-6, momentum=0.5),
                                        Softplus())
        # second hidden layer with 500 neurons in input (output of previous hidden layer) and 500 in output
        self.dense_layer_2 = Sequential(Linear(500,500,bias=False),
                                        BatchNorm1d(500, affine=False, eps=1e-6, momentum=0.5),
                                        Softplus())
        # apply weight norm at output of the model
        self.weight_normalization_ = Sequential(weight_normalization(500,output_dimension),Softplus())
        # weight initialization using kaiming uniform distribution for the first and second dense layers (fully connected layers)
        weight_initialization.kaiming_uniform_(self.dense_layer_1[0].weight)
        weight_initialization.kaiming_uniform_(self.dense_layer_2[0].weight)

    def forward(self,batch_size):
        x = torch.rand(batch_size, self.noise_dimension).to(device=GPU)
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        x = self.weight_normalization_(x)
        return x

