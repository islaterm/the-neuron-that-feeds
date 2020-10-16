from typing import Callable, List, Union

import torch
from torch import Tensor, nn
from torch.nn import Module, Parameter, ParameterList

from activation_functions import relu, sig, softmax

ActivationFunction = Union[Callable[[torch.Tensor, float], Tensor], Callable[[Tensor], Tensor]]


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
        self.__size = len(activation_functions)
        self.__weights = ParameterList(
            [Parameter(torch.rand(neurons[i], neurons[i + 1])) for i in range(0, len(neurons) - 1)])
        self.__biases = ParameterList(
            [Parameter(torch.rand(1, layer_size)) for layer_size in hidden_sizes])
        self.__activation_functions = activation_functions

    def forward(self, nn_input: Tensor) -> Tensor:
        weights = self.__weights
        biases = self.__biases
        functions = self.__activation_functions
        prev_output = nn_input
        for i in range(0, self.__size):
            prev_output = functions[i](torch.matmul(prev_output, weights[i]) + biases[i])
        return softmax(prev_output, dim=-1) ## FIXME: Output dimensions doesn't match

    # region : Utility
    def print_weights(self) -> None:
        """
        Prints a summary of the weights of this network.
        """
        print("Weights:")
        for layer, weight in enumerate(self.weights):
            print(f"\tLayer {layer}: {weight.size()}")

    def print_biases(self) -> None:
        """
        Prints a summary of the biases of this network.
        """
        print("Biases:")
        for layer, bias in enumerate(self.biases):
            print(f"\tLayer {layer}: {bias.size()}")

    def print_activation_functions(self):
        """
        Prints a summary of the biases of this network.
        """
        print("Activation functions:")
        for layer, fun in enumerate(self.activation_functions):
            print(f"\tLayer {layer}: {fun.__name__}")

    # endregion

    # region : Properties
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

    @property
    def activation_functions(self):
        return self.__activation_functions

    @activation_functions.setter
    def activation_functions(self, nn_functions: List[Tensor]):
        self.__activation_functions = nn_functions
    # endregion


if __name__ == '__main__':
    network = FFNN(300, [50, 30], [relu, sig], 10)
    print(network.forward(torch.randn(1, 300)))
