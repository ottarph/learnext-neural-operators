import dolfin as df
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from subdomain.dataset import load_dataset, load_boundary_indices, INTERFACE_TAG
from subdomain.mask import local_mask

def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    DATASET_PATH = Path("dataset/learnext_period_p1")
    dataset, msh, order = load_dataset(DATASET_PATH)
    interface_indices_np = load_boundary_indices(DATASET_PATH, INTERFACE_TAG)
    interface_indices = torch.tensor(interface_indices_np, dtype=torch.int64)


    LEFT_LIM = 0.3
    RIGHT_LIM = 0.9
    omega = df.CompiledSubDomain("x[0] >= LL - tol && x[0] <= RL + tol", LL=LEFT_LIM, RL=RIGHT_LIM, tol=1e-9)
    mask_df = local_mask(msh, omega, 1.0, order=order)


    subdomain_indices_np = np.flatnonzero(mask_df.vector()[:])
    subdomain_indices = torch.tensor(subdomain_indices_np, dtype=torch.int64)


    subdomain_mask = torch.tensor(mask_df.vector()[subdomain_indices_np].reshape(-1,1), dtype=torch.float32)

    dof_locations_np = df.FunctionSpace(msh, "CG", order).tabulate_dof_coordinates()
    subdomain_dof_locations = torch.tensor(dof_locations_np[subdomain_indices_np,:], dtype=torch.float32)


    torch.manual_seed(seed=0)

    from neuraloperators.encoders import IdentityEncoder as Id
    # Id = nn.Identity
    from neuraloperators.deeponet import DeepONet
    from neuraloperators.networks import MLP

    combine_size = 32
    branch_width = 256
    branch_depth = 6
    branch_widths = [len(interface_indices_np)*2] + [branch_width] * branch_depth + [combine_size]
    trunk_width = 256
    trunk_depth = 6
    trunk_widths = [2] + [trunk_width] * trunk_depth + [combine_size]

    branch = MLP(branch_widths, nn.ReLU())
    trunk = MLP(trunk_widths, nn.ReLU())

    deeponet = DeepONet(Id(), branch, Id(), trunk, 2, 2, combine_style=2)
    deeponet.to(device)


    lr = 1e-4
    optimizer = torch.optim.Adam(deeponet.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)


    import torch.utils.data as tud
    batch_size = 128
    dataloader = tud.DataLoader(dataset, shuffle=True, batch_size=batch_size)

    
    trunk_input = subdomain_dof_locations.unsqueeze(0).to(device)
    subdomain_mask_device = subdomain_mask.to(device)
    
    
    losses = []
    epochs = 20_000
    epoch_loop = tqdm(range(1, epochs+1), position=0, desc=f"Epoch #000, loss =  ???   , lr = {lr:.1e}")
    for epoch in epoch_loop:
        print_loss = 0.0
        for harm, biharm in dataloader:
            optimizer.zero_grad()

            branch_input = harm[:,interface_indices,:].flatten(start_dim=-2).to(device)
            harm_sub = harm[:,subdomain_indices,:].to(device)
            biharm_sub = biharm[:,subdomain_indices,:].to(device)

            deeponet_output = deeponet(branch_input, trunk_input)
            correction = deeponet_output * subdomain_mask_device
            pred = harm_sub + correction
            
            loss = loss_fn(pred, biharm_sub)
            loss.backward()
            
            print_loss += loss.item()

            optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        epoch_loop.set_description_str(f"Epoch #{epoch:05}, loss = {print_loss:.2e}, lr = {lr:.1e}")
        losses.append(print_loss)
        scheduler.step()


    fig, ax = plt.subplots()
    ax.semilogy(range(epochs), losses, 'k-')
    fig.savefig("subdomain/output/loss.pdf")
    

    harm, biharm = dataset.tensors[0], dataset.tensors[1]
    branch_input = harm[:,interface_indices,:].flatten(start_dim=-2).to(device)
    harm_sub = harm[:,subdomain_indices,:].to(device)
    biharm_sub = biharm[:,subdomain_indices,:].to(device)

    deeponet_output = deeponet(branch_input, trunk_input)
    correction = deeponet_output * subdomain_mask_device

    displacement = harm.detach()
    displacement[:,subdomain_indices,:] += correction.detach().cpu()

    displacement_np = displacement.detach().numpy()
    biharm_np = biharm.detach().numpy()

    from tools.mesh_quality import MeshQuality
    scaled_jacobian = MeshQuality(msh, "scaled_jacobian")
    min_mesh_mq = np.zeros(displacement_np.shape[0])
    bhmin_mesh_mq = np.zeros(displacement_np.shape[0])

    V = df.VectorFunctionSpace(msh, "CG", order)
    u = df.Function(V)
    scratch = np.zeros(V.dim())
    for k in range(displacement_np.shape[0]):
        uh_k = displacement_np[k,:,:]
        scratch[0::2] = uh_k[:,0]
        scratch[1::2] = uh_k[:,1]
        u.vector()[:] = scratch
        min_mesh_mq[k] = scaled_jacobian(u).min()
        uh_k = biharm_np[k,:,:]
        scratch[0::2] = uh_k[:,0]
        scratch[1::2] = uh_k[:,1]
        u.vector()[:] = scratch
        bhmin_mesh_mq[k] = scaled_jacobian(u).min()


    fig, ax = plt.subplots()
    ax.plot(range(min_mesh_mq.shape[0]), min_mesh_mq, 'k-', label="DeepONet")
    ax.plot(range(min_mesh_mq.shape[0]), bhmin_mesh_mq, 'k:', label="biharmonic")
    ax.set_xlim(xmin=0, xmax=min_mesh_mq.shape[0])
    ax.set_ylim(ymin=0.0, ymax=0.75)
    ax.legend(loc="lower left")
    fig.savefig("subdomain/output/meshquality.pdf")


    with df.XDMFFile("subdomain/output/prediction.xdmf") as outfile:
        outfile.write(msh)
        for k in tqdm(range(displacement_np.shape[0])):
            uh_k = displacement_np[k,:,:]
            scratch[0::2] = uh_k[:,0]
            scratch[1::2] = uh_k[:,1]
            u.vector()[:] = scratch
            outfile.write_checkpoint(u, "uh", k, append=True)

    with df.XDMFFile("subdomain/output/mask.xdmf") as outfile:
        outfile.write(msh)
        outfile.write_checkpoint(mask_df, "lh", 0, append=True)


    return


if __name__ == "__main__":
    main()
