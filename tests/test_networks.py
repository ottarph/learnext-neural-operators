import torch
import torch.nn as nn

from neuraloperators.networks import *

def test_skip_connection():

    mod = MLP([4, 4, 4])
    skip = SkipConnection(mod)

    print(skip(torch.rand(4)))
    print(skip)


    mod = MLP([4, 2, 2, 2, 4])
    skip = SkipConnection(mod)

    print(skip(torch.rand(4)))
    print(skip)


    fully_skip = nn.Sequential(nn.Sequential(nn.Linear(4,4), nn.ReLU()),
                               nn.Sequential(nn.Linear(4,4), nn.ReLU()),
                               nn.Sequential(nn.Linear(4,4), nn.ReLU()),
                               nn.Linear(4,4))
    
    print(fully_skip)
    print(fully_skip(torch.rand(4)))

    return

def test_residual_mlp_block():

    block = ResidualMLPBlock(4, normalization=nn.LayerNorm((4,)))
    print(block)

    print(block(torch.rand((2,4))))

    norm = nn.LayerNorm((20, 2))
    print(norm(torch.rand((20, 2))))

    x0 = torch.rand(2)
    x = torch.cat([x0[None,...]*0.05*i for i in range(1,21)], dim=0)
    print(x)
    print(norm(x) - norm.bias)
    print(norm.bias)
    print(norm.weight)

    norm = nn.LayerNorm((2,))
    print(norm(torch.rand((2,))))

    x0 = torch.rand(2)
    x = torch.cat([x0[None,...]*0.05*i for i in range(1,21)], dim=0)
    print(x)
    print(norm(x) - norm.bias)
    print(norm.bias)
    print(norm.weight)


    return

def test_residual_mlp():
    
    prepro  = MLP([4, 128])
    resnet  = ResidualMLP(128, 4, nn.ReLU(), nn.LayerNorm((128,)))
    postpro = MLP([128, 6])

    model = nn.Sequential(prepro, resnet, postpro)
    
    print(model)

    print(model(torch.rand((2, 10, 4))))

    return


def test_split_additive():

    split_add_model = SplitAdditive(MLP([2, 64]), MLP([2, 64]), 2, 2)

    x = torch.rand((8, 60, 4))
    assert split_add_model(x).shape == (8, 60, 64)

    x = torch.rand((4))
    x1 = torch.cat([x[:2], x[:2]], dim=0)
    x2 = torch.cat([x[:2], x[2:]], dim=0)
    x3 = torch.cat([x[2:], x[:2]], dim=0)
    x4 = torch.cat([x[2:], x[2:]], dim=0)
    y1 = split_add_model(x1)
    y2 = split_add_model(x2)
    y3 = split_add_model(x3)
    y4 = split_add_model(x4)

    # Check split-additivity. If correctly implemented, then
    #             y1 + y4 = sa_1(x[:2]) + sa_1(x[2:]) + sa_2(x[:2]) + sa_2(x[2:])
    #             y2 + y3 = sa_1(x[:2]) + sa_1(x[2:]) + sa_2(x[2:]) + sa_2(x[:2])
    # y1 + y4 - (y2 + y3) = 0
    eps = torch.finfo(torch.get_default_dtype()).eps
    assert torch.norm( y1 + y4 - y2 - y3) < eps * 64

    return


if __name__ == "__main__":
    test_skip_connection()
    test_residual_mlp_block()
    test_residual_mlp()
    test_split_additive()
