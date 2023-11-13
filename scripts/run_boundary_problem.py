import argparse
import torch
import torch.nn as nn
import numpy as np
import shutil

from pathlib import Path
from neuraloperators.loading import load_deeponet_problem
from neuraloperators.training import Context, train_with_dataloader
from dataset.dataset import MeshData

def run_boundary_problem(problem_file: Path, results_dir: Path, 
                         xdmf_overwrite: bool = False, save_xdmf: bool = True,
                         latest_results_dir: Path = Path("results/latest")) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    deeponet, train_dataloader, val_dataloader, dset, \
    optimizer, scheduler, loss_fn, num_epochs, mask_tensor = load_deeponet_problem(problem_file)

    x_data: MeshData = dset.x_data
    y_data: MeshData = dset.y_data

    from neuraloperators.deepsets import DeepSets
    from neuraloperators.networks import ResidualMLP, MLP
    # if isinstance(deeponet.branch, DeepSets) and isinstance(deeponet.branch.representer, MLP):
    #     widths, activation = deeponet.branch.representer.widths, deeponet.branch.representer.activation
    #     deeponet.branch.representer = ResidualMLP(widths, activation)
    # if isinstance(deeponet.branch, DeepSets) and isinstance(deeponet.branch.processor, MLP):
    #     widths, activation = deeponet.branch.processor.widths, deeponet.branch.processor.activation
    #     deeponet.branch.processor = ResidualMLP(widths, activation)
    # if isinstance(deeponet.trunk, MLP):
    #     widths, activation = deeponet.trunk.widths, deeponet.trunk.activation
    #     deeponet.trunk = ResidualMLP(widths, activation)

    # deeponet.branch.representer = nn.Sequential(
    #     MLP([4, 128]),
    #     ResidualMLP(128, 4, nn.ReLU(), nn.LayerNorm((128,)))
    # )
    # deeponet.branch.processor = MLP([128, 128, 32])
    print(deeponet)

    print(sum(map(lambda p: p.numel(), deeponet.branch.parameters())))
    deeponet.branch = nn.Sequential(
        MLP([412, 256]),
        ResidualMLP(256, 4),# normalization=nn.BatchNorm1d((256,))),
        MLP([256, 32])
    )
    print(sum(map(lambda p: p.numel(), deeponet.branch.parameters())))
    # quit()

    print(sum(map(lambda p: p.numel(), deeponet.trunk.parameters())))
    deeponet.trunk = nn.Sequential(
        MLP([2, 256]),
        ResidualMLP(256, 3),# normalization=nn.LayerNorm((256,))),
        MLP([256, 32])
    )
    print(sum(map(lambda p: p.numel(), deeponet.trunk.parameters())))
    print(deeponet)
    # quit()

    evaluation_points = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())

    from neuraloperators.cost_functions import DataInformedLoss
    x0, y0 = next(iter(train_dataloader))
    z0 = x0 + deeponet(x0, evaluation_points) * mask_tensor
    if isinstance(loss_fn, DataInformedLoss):
        loss_fn(x0, y0, z0)
    else:
        loss_fn(z0, y0)


    from neuraloperators.deeponet import DeepONet
    class EvalWrapper(nn.Module):
        def __init__(self, deeponet: DeepONet, evaluation_points: torch.Tensor, mask_tensor: torch.Tensor):
            super().__init__()

            self.deeponet = deeponet

            if len(mask_tensor.shape) == 1:
                mask_tensor = mask_tensor[:,None]

            self.register_buffer("evaluation_points", evaluation_points)
            self.register_buffer("mask_tensor", mask_tensor)
            self.evaluation_points: torch.Tensor
            self.mask_tensor: torch.Tensor
            self.evaluation_points.requires_grad_(False)
            self.mask_tensor.requires_grad_(False)

            return
        
        def forward(self, uh: torch.Tensor) -> torch.Tensor:
            return uh + self.deeponet(uh, self.evaluation_points) * self.mask_tensor

    
    net = EvalWrapper(deeponet, evaluation_points, mask_tensor)
    net.to(device)
    print(net)
    context = Context(net, loss_fn, optimizer, scheduler)

    try:
        train_with_dataloader(context, train_dataloader, num_epochs, device, val_dataloader=val_dataloader)
    except KeyboardInterrupt:
        if input("Training interrupted. Save current progress? (y/n): ").lower() != "y":
            quit()

    results_data_dir = results_dir / "data"
    context.save_results(results_data_dir)
    context.save_summary(results_data_dir)
    context.plot_results(results_dir)
    context.save_model(results_dir)
    shutil.copy(problem_file, results_dir / "problem.json")

    latest_results_data_dir = latest_results_dir / "data"
    context.save_results(latest_results_data_dir)
    context.save_summary(latest_results_data_dir)
    context.plot_results(latest_results_dir)
    context.save_model(latest_results_dir)
    shutil.copy(problem_file, latest_results_dir / "problem.json")


    net.to("cpu")
    x0, y0 = next(iter(train_dataloader))
    x0, y0 = x0[[0],...], y0[[0],...]
    z0 = net(x0)

    if save_xdmf:
        from tools.xdmf_io import pred_to_xdmf

        pred_to_xdmf(net, dset, results_dir / "pred", overwrite=xdmf_overwrite)
        shutil.copy(results_dir / "pred.xdmf", latest_results_dir / "pred.xdmf")
        shutil.copy(results_dir / "pred.h5", latest_results_dir / "pred.h5")

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("problems/default.json"), type=Path)
    parser.add_argument("--results-dir", default=Path("results/default_skiptest"), type=Path)
    # parser.add_argument("--problem-file", default=Path("problems/defaultdeepsets.json"), type=Path)
    # parser.add_argument("--results-dir", default=Path("results/defaultdeepsets_skiptest"), type=Path)
    # parser.add_argument("--problem-file", default=Path("problems/defaultvidon.json"), type=Path)
    # parser.add_argument("--results-dir", default=Path("results/defaultvidon"), type=Path)
    parser.add_argument("--save-xdmf", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    problem_file: Path = args.problem_file
    results_dir: Path = args.results_dir
    save_xdmf: bool = args.save_xdmf
    overwrite: bool = args.overwrite

    if not problem_file.is_file():
        raise RuntimeError("Given problem description file is not a file.")
    if results_dir.exists():
        if len(list(results_dir.iterdir())) > 0 and overwrite == False:
            print(f"Results directory ({str(results_dir)}) is not empty. Continuing might overwrite data.")
            if not input("Continue? (y/n): ").lower() == "y":
                print("\nExiting program.")
                quit()
    else:
        results_dir.mkdir(parents=True)

    run_boundary_problem(problem_file, results_dir, xdmf_overwrite=True, save_xdmf=save_xdmf)

    return


if __name__ == "__main__":
    main()
