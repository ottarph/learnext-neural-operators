import torch
import torch.nn as nn

from typing import Callable, Sequence

class MLP(nn.Module):

    def __init__(self, widths: Sequence[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation

        self.layers = nn.ModuleList([nn.Linear(w1, w2) for (w1, w2) in zip(widths[:-1], widths[1:])])

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers[:-1]:
            y = self.activation(layer(y))
        y = self.layers[-1](y)
        return y
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: # Hack to make type hint for self(x) be tensor
        return super().__call__(x)

class ResidualMLPBlock(nn.Module):

    def __init__(self, width: int, 
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 normalization: Callable[[torch.Tensor], torch.Tensor] = nn.Identity()):
        super().__init__()

        self.width = width
        self.activation = activation
        self.normalization = normalization

        self.linears = nn.ModuleList([nn.Linear(width, width), nn.Linear(width, width)])

        return

    def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Ordering from "Identity Mappings in Deep Residual Networks", fig 1b.
        """
        y = self.linears[0](self.activation(self.normalization(x)))
        z = self.linears[1](self.activation(self.normalization(y)))

        return x + z

class ResidualMLP(nn.Module):

    def __init__(self, width: int, num_blocks: int,
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 normalization: Callable[[torch.Tensor], torch.Tensor] = nn.Identity()):
        super().__init__()

        self.width = width
        self.num_blocks = num_blocks
        self.activation = activation
        self.normalization = normalization
        self.blocks = nn.ModuleList([ResidualMLPBlock(width, activation=activation, normalization=normalization)
                                     for _ in range(num_blocks)])

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = x
        for b in self.blocks:
            y = b(y)

        return y


class SkipConnection(nn.Module):
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)
    

class SplitAdditive(nn.Module):

    def __init__(self, module_1: nn.Module, module_2: nn.Module, length_1: int, length_2: int):
        super().__init__()

        self.module_1 = module_1
        self.module_2 = module_2

        self.length_1 = length_1
        self.length_2 = length_2

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module_1(x[...,:self.length_1]) + self.module_2(x[...,self.length_1:])

# __all__ = ["MLP", "SplitAdditive"]
