import torch
from openfold.np import residue_constants as rc
from typing import List, Union, Tuple, Optional

def get_rotation(tgt, src):
    """
    tgt: (N, 3) tensor
    src: (N, 3) tensor
    """
    assert tgt.shape == src.shape
    assert tgt.shape[1] == 3
    
    tgt = tgt - tgt.mean(0)
    src = src - src.mean(0)
    
    u, s, v = torch.svd(src.t() @ tgt)
    r = u @ v.t()
    x = tgt.mean(0) - r @ src.mean(0)
    
    return r, x


def stru_align(output_list):
    ca_idx = rc.atom_order["CA"]    
    tgt_ca_pos = output_list[0]["final_atom_positions"][:, ca_idx, :]
    tgt_ca_pos.to(dtype=torch.float32)
    for index, out in enumerate(output_list[1:], 1):
        ca_pos = out["final_atom_positions"][ :, ca_idx, :]
        ca_pos.to(dtype=torch.float32)
        r, x = get_rotation(tgt_ca_pos, ca_pos)
        out["final_atom_positions"] = out["final_atom_positions"] @ r + x
    
    return output_list


def get_distance_to_center(stru_list):
    """
    Calculate the center of structures.
    Returns the distance of each structure from the center.
    
    stru_list: list of (N, 37, 3) array
    """
    ca_idx = rc.atom_order["CA"]
    stru_list = [stru[:, ca_idx, :] for stru in stru_list]
    
    assert len(stru_list) > 0
    assert stru_list[0].shape[1] == 3
    
    center = sum(stru.mean(0) for stru in stru_list) / len(stru_list)
    dist_list = [torch.norm(stru - center, dim=1) for stru in stru_list]
    dist_mean_list = [dist.mean() for dist in dist_list]
    
    return dist_mean_list


def calc_rmsd(coord1: torch.Tensor, coord2: torch.Tensor) -> float:
    """
    Calculate RMSD between two sets of coordinates without alignment.
    
    Args:
        coord1: (N, 3) tensor of coordinates
        coord2: (N, 3) tensor of coordinates
    
    Returns:
        float: RMSD value
    """
    assert coord1.shape == coord2.shape
    diff = coord1 - coord2
    return torch.sqrt(torch.mean(torch.sum(diff * diff, dim=-1)))

def calc_aligned_rmsd(coord1: torch.Tensor, 
                     coord2: torch.Tensor, 
                     align: bool = True) -> float:
    """
    Calculate RMSD between two sets of coordinates with optional alignment.
    
    Args:
        coord1: (N, 3) tensor of coordinates
        coord2: (N, 3) tensor of coordinates
        align: whether to align structures before RMSD calculation
    
    Returns:
        float: RMSD value after alignment
    """
    if align:
        r, x = get_rotation(coord1, coord2)
        coord2_aligned = coord2 @ r + x
        return calc_rmsd(coord1, coord2_aligned)
    return calc_rmsd(coord1, coord2)

def calc_pairwise_rmsd(structures: List[torch.Tensor], 
                      atom_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate pairwise RMSD matrix for a list of structures efficiently.
    
    Args:
        structures: list of (N, atoms, 3) tensors containing atom coordinates
        atom_mask: optional (N, atoms) boolean tensor for atom selection
        
    Returns:
        (M, M) tensor containing pairwise RMSD values where M is len(structures)
    """
    n_structs = len(structures)
    if n_structs == 0:
        return torch.zeros((0, 0))
        
    # Convert to same device if needed
    device = structures[0].device
    structures = [s.to(device) for s in structures]
    
    # Apply atom mask if provided
    if atom_mask is not None:
        structures = [s[:, atom_mask, :] for s in structures]
    
    # Preallocate result matrix
    rmsd_matrix = torch.zeros((n_structs, n_structs), device=device)
    
    # Calculate upper triangle
    for i in range(n_structs):
        for j in range(i+1, n_structs):
            rmsd = calc_aligned_rmsd(structures[i], structures[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd  # Matrix is symmetric
            
    return rmsd_matrix


def filter_output(output_list, rmsd_thres=4.0):
    """
    Filter structures based on their pairwise RMSD.
    
    Args:
        output_list: list of output dictionaries from model
        rmsd_thres: RMSD threshold for filtering
    
    Returns:
        list of output dictionaries for filtered structures
    """
    ca_idx = rc.atom_order["CA"] 
    stru_list = [out["final_atom_positions"][:, ca_idx, :] for out in output_list]
    rmsd_matrix = calc_pairwise_rmsd(stru_list)
    
    keep_idx = [i for i in range(len(output_list)) if rmsd_matrix[i, :].mean().item() < rmsd_thres]
    
    return [output_list[i] for i in keep_idx]