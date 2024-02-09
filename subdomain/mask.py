import dolfin as df
import numpy as np

from pathlib import Path

from typing import Any, NewType

Mesh = NewType("Mesh", Any)
def global_mask(msh: Mesh, f_str: str, order: int = 1, normalize: bool = True, expr_degree: int = 5) -> df.Function:

    V = df.FunctionSpace(msh, "CG", order)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Expression(f_str, degree=expr_degree)

    a = df.inner(df.grad(u), df.grad(v)) * df.dx
    l = f * v * df.dx

    bc = df.DirichletBC(V, df.Constant(0.0), "on_boundary")

    A, b = df.assemble_system(a, l, [bc])

    uh = df.Function(V, name="uh")
    df.solve(A, uh.vector(), b)

    if normalize:
        uh.vector()[:] /= uh.vector()[:].max()

    return uh


SubDomain = NewType("SubDomain", Any)
def local_mask(msh: Mesh, omega: SubDomain, f_const: float = 1.0, order: int = 1, normalize: bool = True) -> df.Function:

    SUBDOMAIN_TAG = 1
    SUBDOMAIN_BOUNDARY_TAG = 2

    subdomains = df.MeshFunction("size_t", msh, 2, 0)
    omega.mark(subdomains, SUBDOMAIN_TAG)



    # Mark boundaries of the patch
    tdim = msh.topology().dim()
    msh.init(tdim-1, tdim)
    facet_f = df.MeshFunction("size_t", msh, tdim-1, 0)


    patch = subdomains.array()
    for facet in df.facets(msh):
        marked_cells = patch[facet.entities(tdim)]
        if len(marked_cells) == 1:
            if marked_cells[0] == 1:
                facet_f[facet] = SUBDOMAIN_BOUNDARY_TAG
        elif marked_cells[0] != marked_cells[1]:
            facet_f[facet] = SUBDOMAIN_BOUNDARY_TAG


    dx = df.Measure("dx", domain=msh, subdomain_data=subdomains)

    V = df.FunctionSpace(msh, "CG", order)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = df.inner(df.grad(u), df.grad(v))*dx(SUBDOMAIN_TAG)
    L = df.inner(df.Constant(f_const), v)*dx(SUBDOMAIN_TAG)



    bc = df.DirichletBC(V, df.Constant(0.0), facet_f, SUBDOMAIN_BOUNDARY_TAG)
    bcs = [bc]
    A, b = df.assemble_system(a, L, bcs)
    A.ident_zeros() # Make system nonsingular

    uh = df.Function(V, name="uh")
    df.solve(A, uh.vector(), b)

    if normalize:
        uh.vector()[:] /= uh.vector()[:].max()

    return uh


def main():

    DATASET_PATH = Path("dataset/learnext_period_p1")

    msh_file = df.XDMFFile(str(DATASET_PATH / "input.xdmf"))
    msh = df.Mesh()
    msh_file.read(msh)

    p = 1

    uh = global_mask(msh, "1", order=p)


    df.File("output/fenics/global_mask.pvd") << uh


    LEFT_LIM = 0.4
    RIGHT_LIM = 0.8

    # class Omega(df.SubDomain):
    #     def inside(self, x, on_boundary):
    #         return x[0] >= LEFT_LIM - 1e-9 and x[0] <= RIGHT_LIM + 1e-9
    # omega = Omega()

    omega = df.CompiledSubDomain("x[0] >= LL - tol && x[0] <= RL + tol", LL=LEFT_LIM, RL=RIGHT_LIM, tol=1e-9)

    uh = local_mask(msh, omega, 1.0, order=p)

    df.File("output/fenics/local_mask.pvd") << uh

    print(f"{np.flatnonzero(uh.vector()[:]).shape = }")


    return



if __name__ == "__main__":
    main()
