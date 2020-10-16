import random
from typing import List

import numpy
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from activation_functions import relu
from neural_network import FFNN

ROOT = "https://raw.githubusercontent.com/dccuchile/CC6204/master/2020/tareas/tarea1/mnist_weights"


def load_weights() -> List[Tensor]:
    return [torch.from_numpy(numpy.loadtxt(f"{ROOT}/{filename}.txt")).float() for filename in
            ["W1", "W2", "U"]]


def load_biases() -> List[Tensor]:
    return [torch.from_numpy(numpy.loadtxt(f"{ROOT}/{filename}.txt")).float() for filename in
            ["b1", "b2", "c"]]


if __name__ == '__main__':
    dataset = MNIST('mnist', train=False, transform=ToTensor(), download=True)
    idx = random.randint(0, len(dataset))
    image_tensor, value = dataset[idx]
    input_tensor = torch.flatten(image_tensor)
    network = FFNN(784, [32, 16], [relu, relu], 10)
    network.weights = load_weights()
    network.biases = load_biases()
    network.print_weights()
    network.print_biases()
    torch.set_printoptions(sci_mode=False)
    print(network.forward(input_tensor) * 100)
