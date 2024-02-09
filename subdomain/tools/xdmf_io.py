import dolfin as df
import torch
import numpy as np
import tqdm

from os import PathLike
from pathlib import Path



def find_cg_order_of_xdmf(path_to_xdmf: PathLike) -> int:

    with open(path_to_xdmf, "r") as infile:
        for k in range(40):
            line = infile.readline()
            if "ElementDegree" in line:
                ind = next(filter(lambda i: line[i:i+len("ElementDegree")] == "ElementDegree",
                                  range(len(line))))
                order = int(line[ind + len("ElementDegree") + 2])
                return order
    raise RuntimeError


def find_number_of_checkpoints_of_xdmf(path_to_xdmf: PathLike) -> int:

    with open(path_to_xdmf, "r") as infile:
        lines = infile.readlines()

    for line in lines[:-20:-1]:
        if "Time" in line:
            num = line.split('"')[1]
            num = float(num)
            num = int(num)
            return num + 1
    raise RuntimeError


def convert_dataset(dataset_dir: PathLike) -> None:
    dataset_dir = Path(dataset_dir)

    input_file = df.XDMFFile(str(DATASET_DIR / "input.xdmf"))
    output_file = df.XDMFFile(str(DATASET_DIR / "output.xdmf"))

    assert not ( (DATASET_DIR / "input.npy").exists() or (DATASET_DIR / "output.npy").exists() )

    msh = df.Mesh()
    input_file.read(msh)

    from xdmf_io import find_cg_order_of_xdmf
    order = find_cg_order_of_xdmf(dataset_dir / "input.xdmf")
    V = df.VectorFunctionSpace(msh, "CG", order)
    u_input = df.Function(V)
    u_output = df.Function(V)



    def df_to_np(u: df.Function) -> np.ndarray:

        uh = np.zeros((u.function_space().dim() // 2, 2))
        uh[:,0] = u.vector()[0::2]
        uh[:,1] = u.vector()[1::2]

        return uh

    NUM_SNAPSHOTS = find_number_of_checkpoints_of_xdmf(dataset_dir / "input.xdmf")
    uh_input_arr = np.zeros((NUM_SNAPSHOTS, *df_to_np(u_input).shape))
    uh_output_arr = np.zeros((NUM_SNAPSHOTS, *df_to_np(u_output).shape))
    for k in tqdm.tqdm(range(NUM_SNAPSHOTS)):
        input_file.read_checkpoint(u_input, "uh", k)
        output_file.read_checkpoint(u_output, "uh", k)
        uh_input = df_to_np(u_input)
        uh_output = df_to_np(u_output)
        uh_input_arr[k,...] = uh_input
        uh_output_arr[k,...] = uh_output

    np.save(DATASET_DIR / "input.npy", uh_input_arr)
    np.save(DATASET_DIR / "output.npy", uh_output_arr)

    return


if __name__ == "__main__":
    DATASET_DIR = Path("dataset/learnext_period_p1")
    convert_dataset(DATASET_DIR)
