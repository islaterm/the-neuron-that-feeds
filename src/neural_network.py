from typing import Callable, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import Module, Parameter, ParameterList

from activation_functions import relu, sig

ActivationFunction = Callable[[torch.Tensor, Optional[float]], torch.Tensor]


class FFNN(Module):
    """ Implementation of a Feed Forward Neural Network (FFNN).

        :param size_in:
            the number of neurons in the input layer.
        :param hidden_sizes:
            the list of sizes of the hidden layers.
        :param activation_functions:
            the list of the activation functions.
        :param size_out:
            the number of neurons on the output layer.
    """

    def __init__(self, size_in: int, hidden_sizes: List[int],
                 activation_functions: List[ActivationFunction],
                 size_out: int):
        assert size_out >= 2
        super(FFNN, self).__init__()
        neurons = [size_in]
        neurons.extend(hidden_sizes)
        neurons.append(size_out)
        self.__weights = ParameterList(
            [Parameter(torch.rand(neurons[i], neurons[i + 1])) for i in range(0, len(neurons) - 1)])
        self.__biases = ParameterList(
            [Parameter(torch.rand(layer_size, 0)) for layer_size in hidden_sizes])

    @property
    def weights(self):
        return self.__weights.parameters()

    @weights.setter
    def weights(self, nn_weights: List[Tensor]):
        self.__weights = ParameterList(
            [nn.Parameter(layer_weights) for layer_weights in nn_weights])

    @property
    def biases(self):
        return self.__biases.parameters()

    @biases.setter
    def biases(self, nn_biases: List[Tensor]):
        self.__biases = ParameterList([Parameter(layer_biases) for layer_biases in nn_biases])


if __name__ == '__main__':
    red_neuronal = FFNN(300, [50, 30], [relu, sig], 10)
    for w in red_neuronal.biases:
        print(w)
