from typing import Union

import torch
from torch import nn
import numpy as np
from torch.nn import Sequential

Activation = Union[str, nn.Module]


_str_to_activation = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
}

device = None

def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "relu",
    output_activation: Activation = "identity",
):
    """
    Builds a feedforward neural network

    arguments:
        input_placeholder: placeholder variable for the state (batch_size, input_size)
        scope: variable scope of the network

        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer

    returns:
        output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    mlp = nn.Sequential(*layers)
    mlp.to(device)
    return mlp


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("Using CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(data: Union[np.ndarray, dict], **kwargs):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    else:
        data = torch.from_numpy(data, **kwargs)
        if data.dtype == torch.float64:
            data = data.float()
        return data.to(device)


def to_numpy(tensor: Union[torch.Tensor, dict]):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        return tensor.to("cpu").detach().numpy()

#### MLP class
class Layer(nn.Module):
    def __init__(self, input_size, output_size, activation='relu'):
        super(Layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.layers = nn.ModuleList()

        bias = True
        self.fc = nn.Linear(input_size, output_size, bias=bias)
        self.layers.append(self.fc)

        if self.activation == 'linear':
            self.act_layer = _str_to_activation["identity"]
        else:
            self.act_layer = _str_to_activation[self.activation]
            self.act_layer = self.act_layer()
            self.layers.append(self.act_layer)

        # Initialize the weights
        if bias:
            self.fc.bias.data.fill_(0.0)

        if self.activation != 'identity':
            nn.init.kaiming_uniform_(self.fc.weight, nonlinearity=self.activation)
        else:
            nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x

class DeepFFNN(nn.Module):
    def __init__(self, input_size, output_size, n_layers, size, activation='relu', output_activation='identity'):
        super(DeepFFNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.layers_to_log = [-(i * 2 + 1) for i in range(n_layers + 1)]

        self.layers = nn.ModuleList()

        self.in_layer = Layer(input_size=input_size, output_size=size, activation=self.activation)
        self.layers.extend(self.in_layer.layers)

        self.hidden_layers = []
        for i in range(self.n_layers - 1):
            hidden_layer = Layer(input_size=size, output_size=size, activation=self.activation)
            self.hidden_layers.append(hidden_layer)
            self.layers.extend(self.hidden_layers[i].layers)

        self.out_layer = Layer(input_size=size, output_size=output_size, activation=self.output_activation)
        self.layers.extend(self.out_layer.layers)
        
    def forward(self, x):
        out, _ = self.predict(x)
        return out

    def predict(self, x):
        """
        Forward pass
        :param x: input
        :return: estimated output
        """
        activations = []
        out = self.in_layer.forward(x=x)
        activations.append(out)

        for hidden_layer in self.hidden_layers:
            out = hidden_layer.forward(x=out)
            activations.append(out)

        out = self.out_layer.forward(x=out)
        return out, activations
