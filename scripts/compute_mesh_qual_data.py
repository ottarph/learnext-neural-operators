import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from os import PathLike
from pathlib import Path

from neuraloperators.loading import load_deeponet_problem
from dataset.dataset import load_MeshData, FEniCSDataset, ToDType


def compute_mesh_quality_data(results_dir: PathLike) -> np.ndarray:

    results_dir = Path(results_dir)

    deeponet, _, _, dset, \
    _, _, _, _, mask_tensor = load_deeponet_problem(results_dir / "problem.json")
    eval_points = torch.tensor(dset.x_data.function_space.tabulate_dof_coordinates()[::2,:], dtype=torch.float32)[None,...]

    deeponet.branch.load_state_dict(torch.load(results_dir / "branch.pt", map_location="cpu"))
    deeponet.trunk.load_state_dict(torch.load(results_dir / "trunk.pt", map_location="cpu"))


    from neuraloperators.deeponet import DeepONet
    class EvalWrapper(torch.nn.Module):
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
        
    net = EvalWrapper(deeponet, eval_points, mask_tensor)

    test_dataset_path = Path("dataset/learnext_period_p1")
    test_x_data, test_y_data = load_MeshData(test_dataset_path, style="folders")
    test_dset = FEniCSDataset(test_x_data, test_y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    
    from tools.mesh_quality import mesh_quality_rollout
    mesh_quality_array = mesh_quality_rollout(net, test_dset, 
                                              quality_measure="scaled_jacobian", batch_size=64)
    min_mesh_qual_arr = np.min(mesh_quality_array, axis=1)

    return min_mesh_qual_arr


if __name__ == "__main__":

    # RESULTS_DIR = Path("results/best_run")
    # min_mq_dat = compute_mesh_quality_data(RESULTS_DIR)
    # data_path = Path("hyperparameter_study/best_run/min_mesh_mq.txt")
    # np.savetxt(data_path, min_mq_dat)

    data_path = Path("hyperparameter_study/best_run/min_mesh_mq.txt")
    min_mq_arr = np.loadtxt(data_path)

    from random_initialization.make_plots import make_single_mesh_quality_plot

    figsize = (12,4)
    sp_adj_kws = {"left": 0.05, "top": 0.97, "right": 0.97, "bottom": 0.15}
    fig, ax = make_single_mesh_quality_plot(min_mq_arr, figsize=figsize)
    fig.subplots_adjust(**sp_adj_kws)

    fig.savefig(data_path.with_name("mesh_quality_short.pdf"))

    figsize = (6,3)
    sp_adj_kws = {"left": 0.095, "top": 0.99, "right": 0.99, "bottom": 0.145}
    # sp_adj_kws = {}
    fig, ax = make_single_mesh_quality_plot(min_mq_arr, figsize=figsize)
    fig.subplots_adjust(**sp_adj_kws)
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)

    fig.savefig(data_path.with_name("mesh_quality_squat.pdf"))
