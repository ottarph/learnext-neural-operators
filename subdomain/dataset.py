import torch
import dolfin as df
import numpy as np

import torch.utils.data as tud

from pathlib import Path
from os import PathLike

from typing import NewType, Any, Callable
Mesh = NewType("Mesh", Any)

INTERFACE_TAG = 1
OBSTACLE_TAG = 2

def save_boundary_tags(dataset_path: PathLike) -> None:
    dataset_path = Path(dataset_path)

    msh_file = df.XDMFFile(str(dataset_path / "input.xdmf"))
    msh = df.Mesh()
    msh_file.read(msh)

    facet_f = df.MeshFunction("size_t", msh, 1, 0)

    obstacle_and_interface = df.CompiledSubDomain("on_boundary && (x[0] > 0.02) && (x[0] < 0.8) && (x[1] > 0.02) && (x[1] < 0.4)", tol=1e-9)
    obstacle_and_interface.mark(facet_f, OBSTACLE_TAG)

    interface = df.CompiledSubDomain("on_boundary && (x[1] <= 0.21 + tol) && (x[1] >= 0.19 - tol) && (x[0] > 0.2) && (x[0] < 0.7)", tol=1e-9)
    interface.mark(facet_f, INTERFACE_TAG)


    print(f"{np.count_nonzero(facet_f.array()) = }")
    print(f"{np.count_nonzero(facet_f.array() == INTERFACE_TAG) = }")
    print(f"{np.count_nonzero(facet_f.array() == OBSTACLE_TAG) = }")

    with df.XDMFFile(str(dataset_path / "boundaries.xdmf")) as outfile:
        outfile.write(facet_f)

    return

def load_boundary_indices(dataset_path: PathLike, tag: int = INTERFACE_TAG) -> np.ndarray:
    dataset_path = Path(dataset_path)

    msh_file = df.XDMFFile(str(dataset_path / "input.xdmf"))
    msh = df.Mesh()
    msh_file.read(msh)

    facet_f = df.MeshFunction("size_t", msh, 1, 0)
    with df.XDMFFile(str(dataset_path / "boundaries.xdmf")) as infile:
        infile.read(facet_f)
    
    V = df.FunctionSpace(msh, "CG", 1)
    u = df.Function(V)
    bc = df.DirichletBC(V, df.Constant(1.0), facet_f, tag)
    bc.apply(u.vector())

    inds = np.nonzero(u.vector()[:])

    return inds


def load_dataset(dataset_path: PathLike) -> tuple[tud.Dataset, Mesh, int]:
    dataset_path = Path(dataset_path)

    msh_file = df.XDMFFile(str(dataset_path / "input.xdmf"))
    msh = df.Mesh()
    msh_file.read(msh)

    input_ndarray = np.load(dataset_path / "input.npy")
    output_ndarray = np.load(dataset_path / "output.npy")

    input_tensor = torch.tensor(input_ndarray, dtype=torch.float32)
    output_tensor = torch.tensor(output_ndarray, dtype=torch.float32)

    dataset = tud.TensorDataset(input_tensor, output_tensor)

    from tools.xdmf_io import find_cg_order_of_xdmf
    order = find_cg_order_of_xdmf(dataset_path / "input.xdmf")

    return dataset, msh, order





def main():

    DATASET_PATH = Path("dataset/learnext_period_p1")

    dataset, msh, order = load_dataset(DATASET_PATH)

    print(dataset)
    print(f"{msh.num_vertices() = }")
    print(f"{order = }")

    dataloader = tud.DataLoader(dataset, shuffle=True, batch_size=64)

    x, y = next(iter(dataloader))
    print(f"{x.shape = }")
    print()
    print(f"{y.shape = }")
    print()

    save_boundary_tags(DATASET_PATH)
    print()
    inds_interface = load_boundary_indices(DATASET_PATH, INTERFACE_TAG)
    inds_obstacle = load_boundary_indices(DATASET_PATH, OBSTACLE_TAG)
    print(f"{np.count_nonzero(inds_interface) = }")
    print(f"{np.count_nonzero(inds_obstacle) = }")

    return

if __name__ == "__main__":
    main()
