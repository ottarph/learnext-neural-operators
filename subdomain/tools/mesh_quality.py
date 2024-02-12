import dolfin as df
import numpy as np
import torch
import torch.nn as nn
import pyvista as pv

from typing import NewType, Any
Mesh = NewType("Mesh", Any)

class MeshQuality:

    def __init__(self, mesh: Mesh, quality_measure: str = "scaled_jacobian"):

        self.mesh = mesh
        self.quality_measure = quality_measure

        self.polydata = self.build_polydata(mesh)

        return
    
    def build_polydata(self, mesh: df.Mesh) -> pv.PolyData:

        points = np.column_stack((mesh.coordinates()[:,0], mesh.coordinates()[:,1], np.zeros_like(mesh.coordinates()[:,0])))
        faces = np.concatenate((3*np.ones((mesh.num_cells(), 1), dtype=np.uint32), mesh.cells()), axis=1).flatten()

        return pv.PolyData(points, faces)
    
    def convert_vector_field(self, u: df.Function | np.ndarray | torch.Tensor) -> np.ndarray:
        assert isinstance(u, df.Function | np.ndarray | torch.Tensor)

        if isinstance(u, df.Function):
            assert u.function_space().ufl_element().value_shape() == (2,)
            assert u.function_space().mesh() == self.mesh
            uh_tmp = u.compute_vertex_values()
            uh = np.column_stack((uh_tmp[:len(uh_tmp)//2], uh_tmp[len(uh_tmp)//2:], np.zeros(len(uh_tmp)//2)))

        elif isinstance(u, torch.Tensor):
            assert len(u.shape) == 2
            assert u.shape[-1] == 2
            uh_tmp: np.ndarray = u.detach().numpy().astype(np.int64)
            uh = np.column_stack((uh_tmp[:,0], uh_tmp[:,1], np.zeros_like(uh_tmp[:,0])))

        else:
            assert len(u.shape) == 2
            assert u.shape[-1] == 2
            uh = np.column_stack((u[:,0], u[:,1], np.zeros_like(u[:,0])))

        return uh
    
    def __call__(self, u: df.Function | np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Compute the mesh quality of `self.mesh` deformed by u. If `np.ndarray`- or `torch.Tensor`- inputs are used,
        the user must ensure that these are evaluations at mesh vertices in the correct ordering, same as would be
        given by u.compute_vertex_values().

        Args:
            u (df.Function | np.ndarray | torch.Tensor): Function to deform mesh by. If u

        Returns:
            np.ndarray: The mesh quality of all cells in deformed mesh, ordered the same as self.mesh.cells().
        """

        self.polydata["uh"] = self.convert_vector_field(u)
        warped = self.polydata.warp_by_vector("uh")
        quality = warped.compute_cell_quality(quality_measure=self.quality_measure)
        
        return np.copy(quality.cell_data["CellQuality"])
    
