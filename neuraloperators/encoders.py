import torch
import torch.nn as nn
import dolfin as df
import numpy as np
from os import PathLike
from pathlib import Path

# from dataset.dataset import MeshData
from typing import NewType, Any
MeshData = NewType("MeshData", Any)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor: # Hack to make type hint for self(u, y) be tensor
        return super().__call__(x)

class SequentialEncoder(Encoder):
    def __init__(self, *args):
        super().__init__()

        self.encoders = nn.Sequential(*args)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.encoders(x)
    
class IdentityEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x)

class FlattenEncoder(Encoder):
    def __init__(self, start_dim: int):
        """

        """
        super().__init__()

        self.register_buffer("start_dim", torch.tensor(start_dim, dtype=torch.long))
        self.start_dim: torch.LongTensor

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.flatten(x, start_dim=self.start_dim)
    
    def __repr__(self) -> str:
        return f"FlattenEncoder(start_dim={self.start_dim.item()})"
    
class SplitAdditiveEncoder(Encoder):

    def __init__(self, encoder_1: nn.Module, encoder_2: nn.Module, length_1: int, length_2: int):
        super().__init__()

        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2

        self.length_1 = length_1
        self.length_2 = length_2

        return
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_1(x[...,:self.length_1]) + self.encoder_2(x[...,self.length_1:])


class TensorPrependEncoder(Encoder):
    """ Inserts `tensor` before `x` in the last dimension. """
    
    def __init__(self, tensor: torch.Tensor):
        super().__init__()

        self.register_buffer("tensor", tensor)
        self.tensor: torch.Tensor
        self.tensor.requires_grad_(False)
    
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, "Only works with batched tensors"
        return torch.cat((self.tensor.expand((x.shape[0], -1, -1)), x), dim=-1)


class CoordinateInsertEncoder(TensorPrependEncoder):
    
    def __init__(self, mesh_data: MeshData):
        
        coordinates = mesh_data.dof_coordinates.to(torch.get_default_dtype())
        super().__init__(coordinates)
        self.coordinates: torch.Tensor = self.tensor

        return


class FilterEncoder(Encoder):

    def __init__(self, dim: int, unit_shape_length: int):
        super().__init__()

        self.register_buffer("dim", torch.tensor(dim, dtype=torch.long))
        self.dim: torch.LongTensor

        self.register_buffer("unit_shape_length", torch.tensor(unit_shape_length, dtype=torch.long))
        self.unit_shape_length: torch.LongTensor

        self.select_batch = torch.vmap(self.select_single, randomness="different")
        self.select_batch_double = torch.vmap(self.select_batch, randomness="different")

        return

    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        raise NotImplementedError()

    def select_single(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, self.dim, self.filter(x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == self.unit_shape_length:
            out = self.select_single(x)
        elif len(x.shape) == self.unit_shape_length + 1:
            out = self.select_batch(x)
        elif len(x.shape) == self.unit_shape_length + 2:
            out = self.select_batch_double(x)
        else:
            raise ValueError
        return out
    

class FixedFilterEncoder(FilterEncoder):

    def __init__(self, filter: torch.LongTensor, dim: int, unit_shape_length: int):
        super().__init__(dim=dim, unit_shape_length=unit_shape_length)
        self.register_buffer("filter_tensor", filter)
        self.filter_tensor: torch.LongTensor

        return
    
    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        return self.filter_tensor
    

class BoundaryFilterEncoder(FixedFilterEncoder):

    def __init__(self, mesh_data: MeshData):
        indices = mesh_data.boundary_dofs

        # Assumes only vector ``MeshData``
        super().__init__(filter=indices, dim=-2, unit_shape_length=2)
        
        return
    

class InnerBoundaryFilterEncoder(FixedFilterEncoder):

    def __init__(self, mesh_data: MeshData):
        import dolfin as df
        import numpy as np

        def inner_boundary(x, on_boundary):
            if on_boundary:
                eps = 1e-3
                if df.near(x[1], 0, eps) or df.near(x[1], 0.41, eps) \
                    or df.near(x[0], 0, eps) or df.near(x[0], 2.5, eps):
                    return False
                else:
                    return True
            else:
                return False

        V = df.FunctionSpace(mesh_data.function_space.mesh(), "CG", mesh_data.function_space.ufl_element().degree())
        u = df.Function(V)

        bc = df.DirichletBC(V, df.Constant(1), inner_boundary)
        bc.apply(u.vector())

        ids = np.flatnonzero(u.vector().get_local())

        dof_indices = torch.tensor(ids)

        # Assumes only vector ``MeshData``
        super().__init__(filter=dof_indices, dim=-2, unit_shape_length=2)
        
        return


class ExtractBoundaryDofEncoder(FixedFilterEncoder):

    def __init__(self, fspace: df.FunctionSpace, dof_coords_file_path: PathLike):
        """
            Encoder which reorders dofs to match a different mesh. Necessary if dofs need to be input in a specific order
            and the training was done on a different mesh than the prediction, but matching on the inner boundary.
            `fspace` is the `df.FunctionSpace` of the fluid domain where you compute mesh motion.
            `dof_coords_file_path` is the path to a file containing the coordinates of the inner boundary dofs
            in the memory order for a given mesh. This allows to map inner boundary dofs between different meshes,
            as long as the meshes are "essentially the same" on the inner boundary, i.e. equal up to numerical artifacts.
        """
        dof_coords_file_path = Path(dof_coords_file_path)
        if not dof_coords_file_path.suffix in [".txt", ".npz"]:
            raise ValueError
        
        def inner_boundary(x, on_boundary):
            if on_boundary:
                eps = 1e-3
                if df.near(x[1], 0, eps) or df.near(x[1], 0.41, eps) \
                    or df.near(x[0], 0, eps) or df.near(x[0], 2.5, eps):
                    return False
                else:
                    return True
            else:
                return False
            
        V_scal = df.FunctionSpace(fspace.mesh(), "CG", fspace.ufl_element().degree())

        u = df.Function(V_scal)
        bc = df.DirichletBC(V_scal, df.Constant(1), inner_boundary)
        u.vector()[:] = 0.0
        bc.apply(u.vector())

        int_bnd_ids = np.flatnonzero(u.vector().get_local())
        int_bnd_dof_coords = V_scal.tabulate_dof_coordinates()[int_bnd_ids]

        other_int_bnd_dof_coords = np.loadtxt(dof_coords_file_path) if dof_coords_file_path.suffix == ".txt" else np.load(dof_coords_file_path)

        inds = np.zeros(other_int_bnd_dof_coords.shape[0], dtype=int)

        for i in range(other_int_bnd_dof_coords.shape[0]):
            j = np.argmin( np.linalg.norm( int_bnd_dof_coords - other_int_bnd_dof_coords[i] , axis=1 ) , axis=0 )
            inds[i] = int_bnd_ids[j]

        assert np.linalg.norm(other_int_bnd_dof_coords - V_scal.tabulate_dof_coordinates()[inds]) < 1e-7

        indices = torch.tensor(inds)

        # Assumes only vector ``df.FunctionSpace``
        super().__init__(filter=indices, dim=-2, unit_shape_length=2)

        return
    

class RandomPermuteEncoder(FilterEncoder):

    def __init__(self, dim: int, unit_shape_length: int):
        super().__init__(dim=dim, unit_shape_length=unit_shape_length)

    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.randperm(x.shape[self.dim], device=x.device)
    
    def __repr__(self) -> str:
        return f"RandomPermuteEncoder(dim={self.dim.item()})"
    
    
class RandomSelectEncoder(FilterEncoder):

    def __init__(self, dim: int, unit_shape_length: int, num_inds: int):
        super().__init__(dim=dim, unit_shape_length=unit_shape_length)
        self.register_buffer("num_inds", torch.tensor(num_inds, dtype=torch.long))
        self.num_inds: torch.LongTensor

        return

    def filter(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.randperm(x.shape[self.dim], device=x.device)[:self.num_inds]
    
    def __repr__(self) -> str:
        return f"RandomSelectEncoder(dim={self.dim.item()}, num_inds={self.num_inds.item()})"

