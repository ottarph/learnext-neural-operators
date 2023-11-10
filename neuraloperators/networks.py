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


# class ResidualMLP(nn.Module):

#     # def __init__(self, in_size: int, hidden_size: int, out_size: int, num_layers: int, activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
#     def __init__(self, widths: Sequence[int], skips: Sequence[tuple[int, int]], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
#         super().__init__()

#         # self.widths = [in_size] + [hidden_size] * (num_layers - 2) + [out_size]
#         # self.activation = activation
        
#         # self.layers = nn.ModuleList([nn.Linear(w1, w2) for (w1, w2) in zip(self.widths[:-1], self.widths[1:])])
        
#         self.widths = widths
#         self.skips = skips
#         self.activation = activation

#         self.layers = nn.ModuleList([nn.Linear(w1, w2) for (w1, w2) in zip(self.widths[:-1], self.widths[1:])])
        
#         return
    
#     def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         # keep_inds = {ij[0] for ij in self.skips}
#         # keeps = 
#         keeps = set()
#         for k, layer in enumerate(self.layers):
#             if k in map(lambda ij: ij[0], self.skips):
#                 keeps[k] = x

#             x

#         return

# class ResidualMLP(nn.Module):
#     """
#         MLP, but with skip connection between the first and last equally sized layer.
#     """

#     def __init__(self, in_size: int, hidden_size: int, out_size: int, num_layers: int, activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
#         super().__init__()

        
#         self.widths = [in_size] + [hidden_size] * (num_layers - 2) + [out_size]
#         self.activation = activation

#         layers = nn.ModuleList([nn.Linear(w1, w2) for (w1, w2) in zip(self.widths[:-1], self.widths[1:])])

#         if in_size != hidden_size:
#             self.pre_layers = layers[:1]
#             layers.pop(0)
#         else:
#             self.pre_layers = layers[:0]

#         # if out_size != hidden_size:
#         #     self.post_layers = layers[-1:]
#         #     layers.pop(-1)
#         # else:
#         #     self.post_layers = layers[:0]

#         # self.hidden_layers = layers

#         if out_size != hidden_size:
#             self.hidden_layers = layers[:-1]
#             self.post_layers = layers[-1:]
#         else:
#             self.hidden_layers = layers
#             self.post_layers = layers[:0]


#         return
    
#     def __call__(self, x: torch.Tensor) -> torch.Tensor: return super().__call__(x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         from functools import reduce

#         y = x
#         print(f"{y = }")
#         y = reduce(lambda t, l: self.activation(l(t)), self.pre_layers, y)
#         print(f"{y = }")
#         y = y + reduce(lambda t, l: self.activation(l(t)), self.hidden_layers, y)
#         print(f"{y = }")
#         y = reduce(lambda t, l: l(t), self.post_layers, y)
#         print(f"{y = }")

#         return y


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
