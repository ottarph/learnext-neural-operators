import torch
import torch.nn as nn

from neuraloperators.networks import *


# def test_residual_mlp():

#     res_mlp = ResidualMLP(2, 4, 3, 5)
#     print(res_mlp)

#     z = res_mlp(torch.rand(2))

#     return

def test_skip_connection():

    mod = MLP([4, 4, 4])
    skip = SkipConnection(mod)

    print(skip(torch.rand(4)))
    print(skip)


    mod = MLP([4, 2, 2, 2, 4])
    skip = SkipConnection(mod)

    print(skip(torch.rand(4)))
    print(skip)


    fully_skip = nn.Sequential(nn.Sequential())

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
    # test_residual_mlp()
    test_skip_connection()
    test_split_additive()
