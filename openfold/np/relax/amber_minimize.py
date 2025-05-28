# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Restrained Amber Minimization of a structure."""

import io
import re
import time
from typing import Collection, Optional, Sequence

from absl import logging
from openfold.np import (
    protein,
    residue_constants,
)
import openfold.utils.loss as loss
from openfold.np.relax import cleanup, utils
import ml_collections
import numpy as np
try:
    # openmm >= 7.6
    import openmm
    from openmm import unit
    from openmm import app as openmm_app
    from openmm.app.internal.pdbstructure import PdbStructure
except ImportError:
    # openmm < 7.6 (requires DeepMind patch)
    from simtk import openmm
    from simtk import unit
    from simtk.openmm import app as openmm_app
    from simtk.openmm.app.internal.pdbstructure import PdbStructure

from openfold.scripts.check import AmideBondFixer

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms


def will_restrain(atom: openmm_app.Atom, rset: str) -> bool:
    """Returns True if the atom will be restrained by the given restraint set."""

    if rset == "non_hydrogen":
        return atom.element.name != "hydrogen"
    elif rset == "c_alpha":
        return atom.name == "CA"


def _add_restraints(
    system: openmm.System,
    reference_pdb: openmm_app.PDBFile,
    stiffness: unit.Unit,
    rset: str,
    exclude_residues: Sequence[int],
):
    """Adds a harmonic potential that restrains the system to a structure."""
    assert rset in ["non_hydrogen", "c_alpha"]

    force = openmm.CustomExternalForce(
        "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for i, atom in enumerate(reference_pdb.topology.atoms()):
        if atom.residue.index in exclude_residues:
            continue
        if will_restrain(atom, rset):
            force.addParticle(i, reference_pdb.positions[i])
    logging.info(
        "Restraining %d / %d particles.",
        force.getNumParticles(),
        system.getNumParticles(),
    )
    system.addForce(force)


def chirality_fixer(simulation):
    topology = simulation.topology
    positions = simulation.context.getState(getPositions=True).getPositions()

    d_stereoisomers = []
    for residue in topology.residues():
        if residue.name == "GLY":
            continue

        atom_indices = {atom.name:atom.index for atom in residue.atoms() if atom.name in ["N", "CA", "C", "CB"]}
        vectors = [positions[atom_indices[i]] - positions[atom_indices["CA"]] for i in ["N", "C", "CB"]]

        if np.dot(np.cross(vectors[0], vectors[1]), vectors[2]) < .0*LENGTH**3:
            # If it is a D-stereoisomer then flip its H atom
            indices = {x.name:x.index for x in residue.atoms() if x.name in ["HA", "CA"]}
            positions[indices["HA"]] = 2*positions[indices["CA"]] - positions[indices["HA"]]

            # Fix the H atom in place
            particle_mass = simulation.system.getParticleMass(indices["HA"])
            # Setting the mass of the hydrogen atom to an extremely small value,
            # setting it to 0 will cause an error.
            simulation.system.setParticleMass(indices["HA"], 0.001 * unit.amu)
            d_stereoisomers.append((indices["HA"], particle_mass))

    if len(d_stereoisomers) > 0:
        simulation.context.setPositions(positions)

        # Minimize the energy with the evil hydrogens fixed
        simulation.minimizeEnergy()

        # Minimize the energy letting the hydrogens move
        for atom in d_stereoisomers:
            simulation.system.setParticleMass(*atom)
        simulation.minimizeEnergy()

    return simulation


def _openmm_minimize(
    pdb_str: str,
    max_iterations: int,
    tolerance: unit.Unit,
    stiffness: unit.Unit,
    restraint_set: str,
    exclude_residues: Sequence[int],
    use_gpu: bool,
):
    """Minimize energy via openmm."""

    pdb_file = io.StringIO(pdb_str)
    pdb = openmm_app.PDBFile(pdb_file)

    force_field = openmm_app.ForceField("amber99sb.xml")
    constraints = openmm_app.HBonds
    system = force_field.createSystem(pdb.topology, constraints=constraints)
    
    # remove constraints involving massless particles
    for i in reversed(range(system.getNumConstraints())):
        p1, p2, d = system.getConstraintParameters(i)
        if system.getParticleMass(p1) == 0 or system.getParticleMass(p2) == 0:
            system.removeConstraint(i)
        
    if stiffness > 0 * ENERGY / (LENGTH ** 2):
        _add_restraints(system, pdb, stiffness, restraint_set, exclude_residues)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    simulation = openmm_app.Simulation(
        pdb.topology, system, integrator, platform
    )
    simulation.context.setPositions(pdb.positions)

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance) # TODO: Is need to use this function before chirality_fixer?
    simulation = chirality_fixer(simulation)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    ret["min_pdb"] = _get_pdb_string(simulation.topology, state.getPositions())
    return ret


def _openmm_minimize_select(
    pdb_str: str,
    use_gpu: bool,
    max_iterations: int,
    tolerance: unit.Unit,
):
    """Minimize energy via openmm."""

    pdb_file = io.StringIO(pdb_str)
    pdb = openmm_app.PDBFile(pdb_file)
    force_field = openmm_app.ForceField("amber99sb.xml")
    system = force_field.createSystem(pdb.topology)
    
    # Fix atom
    backbone_atoms = {"N", "CA", "C", "O", "HA"}
    original_masses = {}  # 存储原始质量

    # Set the mass of the backbone atoms to 0 to fix them in place
    for atom in pdb.topology.atoms():
        original_masses[atom.index] = system.getParticleMass(atom.index)
        if atom.name in backbone_atoms:
            system.setParticleMass(atom.index, 0.0 * unit.dalton)

    integrator = openmm.LangevinIntegrator(
        0*unit.kelvin,
        1.0/unit.picosecond,
        1.0*unit.femtosecond
    )
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    simulation = openmm_app.Simulation(
        pdb.topology, system, integrator, platform
    )
    simulation.context.setPositions(pdb.positions)

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    
    simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance)
    for atom_index, mass in original_masses.items():
        simulation.system.setParticleMass(atom_index, mass)
    simulation = chirality_fixer(simulation)

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    ret["min_pdb"] = _get_pdb_string(simulation.topology, state.getPositions())
    
    return ret


def _get_pdb_string(topology: openmm_app.Topology, positions: unit.Quantity):
    """Returns a pdb string provided OpenMM topology and positions."""
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)
        return f.getvalue()


def _check_cleaned_atoms(pdb_cleaned_string: str, pdb_ref_string: str):
    """Checks that no atom positions have been altered by cleaning."""
    cleaned = openmm_app.PDBFile(io.StringIO(pdb_cleaned_string))
    reference = openmm_app.PDBFile(io.StringIO(pdb_ref_string))

    cl_xyz = np.array(cleaned.getPositions().value_in_unit(LENGTH))
    ref_xyz = np.array(reference.getPositions().value_in_unit(LENGTH))

    for ref_res, cl_res in zip(
        reference.topology.residues(), cleaned.topology.residues()
    ):
        assert ref_res.name == cl_res.name
        for rat in ref_res.atoms():
            for cat in cl_res.atoms():
                if cat.name == rat.name:
                    if not np.array_equal(
                        cl_xyz[cat.index], ref_xyz[rat.index]
                    ):
                        raise ValueError(
                            f"Coordinates of cleaned atom {cat} do not match "
                            f"coordinates of reference atom {rat}."
                        )


def _check_residues_are_well_defined(prot: protein.Protein):
    """Checks that all residues contain non-empty atom sets."""
    if (prot.atom_mask.sum(axis=-1) == 0).any():
        raise ValueError(
            "Amber minimization can only be performed on proteins with"
            " well-defined residues. This protein contains at least"
            " one residue with no atoms."
        )


def _check_atom_mask_is_ideal(prot):
    """Sanity-check the atom mask is ideal, up to a possible OXT."""
    atom_mask = prot.atom_mask
    ideal_atom_mask = protein.ideal_atom_mask(prot)
    utils.assert_equal_nonterminal_atom_types(atom_mask, ideal_atom_mask)


def clean_protein(prot: protein.Protein, checks: bool = True):
    """Adds missing atoms to Protein instance.

    Args:
      prot: A `protein.Protein` instance.
      checks: A `bool` specifying whether to add additional checks to the cleaning
        process.

    Returns:
      pdb_string: A string of the cleaned protein.
    """
    _check_atom_mask_is_ideal(prot)

    # Clean pdb.
    prot_pdb_string = protein.to_pdb(prot)
    pdb_file = io.StringIO(prot_pdb_string)
    alterations_info = {}
    fixed_pdb = cleanup.fix_pdb(pdb_file, alterations_info)
    fixed_pdb_file = io.StringIO(fixed_pdb)
    pdb_structure = PdbStructure(fixed_pdb_file)
    cleanup.clean_structure(pdb_structure, alterations_info)

    logging.info("alterations info: %s", alterations_info)

    # Write pdb file of cleaned structure.
    as_file = openmm_app.PDBFile(pdb_structure)
    pdb_string = _get_pdb_string(as_file.getTopology(), as_file.getPositions())
    if checks:
        _check_cleaned_atoms(pdb_string, prot_pdb_string)
    
    headers = protein.get_pdb_headers(prot)    
    if(len(headers) > 0):
        pdb_string = '\n'.join(['\n'.join(headers), pdb_string])
    
    return pdb_string


def make_atom14_positions(prot):
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]
        ]

        restype_atom14_to_atom37.append(
            [
                (residue_constants.atom_order[name] if name else 0)
                for name in atom_names
            ]
        )

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in residue_constants.atom_types
            ]
        )

        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = np.array(
        restype_atom14_to_atom37, dtype=np.int32
    )
    restype_atom37_to_atom14 = np.array(
        restype_atom37_to_atom14, dtype=np.int32
    )
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein.
    residx_atom14_to_atom37 = restype_atom14_to_atom37[prot["aatype"]]
    residx_atom14_mask = restype_atom14_mask[prot["aatype"]]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        prot["all_atom_mask"], residx_atom14_to_atom37, axis=1
    ).astype(np.float32)

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
        np.take_along_axis(
            prot["all_atom_positions"],
            residx_atom14_to_atom37[..., None],
            axis=1,
        )
    )

    prot["atom14_atom_exists"] = residx_atom14_mask
    prot["atom14_gt_exists"] = residx_atom14_gt_mask
    prot["atom14_gt_positions"] = residx_atom14_gt_positions

    prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37.astype(np.int64)

    # Create the gather indices for mapping back.
    residx_atom37_to_atom14 = restype_atom37_to_atom14[prot["aatype"]]
    prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14.astype(np.int64)

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[prot["aatype"]]
    prot["atom37_atom_exists"] = residx_atom37_mask

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res]
        for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack(
        [all_matrices[restype] for restype in restype_3]
    )

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[prot["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = np.einsum(
        "rac,rab->rbc", residx_atom14_gt_positions, renaming_transform
    )
    prot["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = np.einsum(
        "ra,rab->rb", residx_atom14_gt_mask, renaming_transform
    )

    prot["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]
            ]
            atom_idx1 = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(atom_name1)
            atom_idx2 = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    prot["atom14_atom_is_ambiguous"] = restype_atom14_is_ambiguous[
        prot["aatype"]
    ]

    return prot


def find_violations(prot_np: protein.Protein):
    """Analyzes a protein and returns structural violation information.

    Args:
      prot_np: A protein.

    Returns:
      violations: A `dict` of structure components with structural violations.
      violation_metrics: A `dict` of violation metrics.
    """
    batch = {
        "aatype": prot_np.aatype,
        "all_atom_positions": prot_np.atom_positions.astype(np.float32),
        "all_atom_mask": prot_np.atom_mask.astype(np.float32),
        "residue_index": prot_np.residue_index,
    }

    batch["seq_mask"] = np.ones_like(batch["aatype"], np.float32)
    batch = make_atom14_positions(batch)

    violations = loss.find_structural_violations_np(
        batch=batch,
        atom14_pred_positions=batch["atom14_gt_positions"],
        config=ml_collections.ConfigDict(
            {
                "violation_tolerance_factor": 12,  # Taken from model config.
                "clash_overlap_tolerance": 1.5,  # Taken from model config.
            }
        ),
    )
    violation_metrics = loss.compute_violation_metrics_np(
        batch=batch,
        atom14_pred_positions=batch["atom14_gt_positions"],
        violations=violations,
    )

    return violations, violation_metrics


def get_violation_metrics(prot: protein.Protein):
    """Computes violation and alignment metrics."""
    structural_violations, struct_metrics = find_violations(prot)
    violation_idx = np.flatnonzero(
        structural_violations["total_per_residue_violations_mask"]
    )

    struct_metrics["residue_violations"] = violation_idx
    struct_metrics["num_residue_violations"] = len(violation_idx)
    struct_metrics["structural_violations"] = structural_violations
    return struct_metrics


def _run_one_iteration(
    *,
    pdb_string: str,
    max_iterations: int,
    tolerance: float,
    stiffness: float,
    restraint_set: str,
    max_attempts: int,
    exclude_residues: Optional[Collection[int]] = None,
    mobile_residue_indices: Optional[Collection[int]] = None,
    use_gpu: bool,
):
    """Runs the minimization pipeline.

    Args:
      pdb_string: A pdb string.
      max_iterations: An `int` specifying the maximum number of L-BFGS iterations.
      A value of 0 specifies no limit.
      tolerance: kcal/mol, the energy tolerance of L-BFGS.
      stiffness: kcal/mol A**2, spring constant of heavy atom restraining
        potential.
      restraint_set: The set of atoms to restrain.
      max_attempts: The maximum number of minimization attempts.
      exclude_residues: An optional list of zero-indexed residues to exclude from
          restraints.
      mobile_residue_indices: An optional list of zero-indexed residues to restrain.
      use_gpu: Whether to run relaxation on GPU
    Returns:
      A `dict` of minimization info.
    """
    exclude_residues = exclude_residues or []

    # Assign physical dimensions.
    tolerance = tolerance * ENERGY
    stiffness = stiffness * ENERGY / (LENGTH ** 2)

    start = time.perf_counter()
    minimized = False
    attempts = 0
    while not minimized and attempts < max_attempts:
        attempts += 1
        try:
            logging.info(
                "Minimizing protein, attempt %d of %d.", attempts, max_attempts
            )
            if mobile_residue_indices is not None:
                ret = _openmm_minimize_select(
                    pdb_string,
                    use_gpu=use_gpu,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                )
            else:
                ret = _openmm_minimize(
                    pdb_string,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    stiffness=stiffness,
                    restraint_set=restraint_set,
                    exclude_residues=exclude_residues,
                    use_gpu=use_gpu,
                )
            minimized = True
        except Exception as e:  # pylint: disable=broad-except
            print(e)
            logging.info(e)
    if not minimized:
        # raise ValueError(f"Minimization failed after {max_attempts} attempts.")
        print(f"Minimization failed after {max_attempts} attempts.")
    ret["opt_time"] = time.perf_counter() - start
    ret["min_attempts"] = attempts
    return ret


def process_ndarray(arr):
    """
    合并 ndarray 的前两个维度，并删除 xyz 全为 0 的数据。

    Args:
    arr: 输入的 ndarray，形状为 [N_seq, N_dim, xyz]。

    Returns:
    处理后的 ndarray，形状为 [N_seq * N_dim, xyz]，其中 xyz 全为 0 的数据已被删除。
    """

    # 1. 合并前两个维度
    arr_reshaped = arr.reshape(-1, arr.shape[-1])

    # 2. 删除 xyz 全为 0 的数据

    # 方法一：使用 np.all 和布尔索引
    mask = ~np.all(arr_reshaped == 0, axis=1)
    arr_filtered = arr_reshaped[mask]


    # 方法二：使用 np.any 和布尔索引 (如果xyz全0，则该行sum为0)
    #  mask = np.any(arr_reshaped != 0, axis=1)  # 查找至少有一个非零元素的行
    #  arr_filtered = arr_reshaped[mask]

    return arr_filtered


def get_atom_positions(pdb_string: str):
    """
    从 PDB 格式字符串中提取所有 ATOM 记录的原子坐标，并返回 NumPy ndarray。

    Args:
      pdb_string: PDB 格式的字符串。

    Returns:
      一个 NumPy ndarray，其中每一行包含一个原子的 x, y, z 坐标。
      如果找不到 ATOM 记录，则返回一个空的 NumPy ndarray。
    """

    atom_lines = re.findall(r"^ATOM\s+.*$", pdb_string, re.MULTILINE)

    if not atom_lines:
      return np.array([])

    coordinates = []
    for line in atom_lines:
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coordinates.append([x, y, z])
        except ValueError:
            continue  # 跳过无法解析的行

    return np.array(coordinates)



def run_pipeline(
    prot: protein.Protein,
    stiffness: float,
    use_gpu: bool,
    max_outer_iterations: int = 1,
    place_hydrogens_every_iteration: bool = True,
    max_iterations: int = 0,
    tolerance: float = 2.39,
    restraint_set: str = "non_hydrogen",
    max_attempts: int = 100,
    checks: bool = True,
    exclude_residues: Optional[Sequence[int]] = None,
):
    """Run iterative amber relax.

    Successive relax iterations are performed until all violations have been
    resolved. Each iteration involves a restrained Amber minimization, with
    restraint exclusions determined by violation-participating residues.

    Args:
      prot: A protein to be relaxed.
      stiffness: kcal/mol A**2, the restraint stiffness.
      use_gpu: Whether to run on GPU
      max_outer_iterations: The maximum number of iterative minimization.
      place_hydrogens_every_iteration: Whether hydrogens are re-initialized
          prior to every minimization.
      max_iterations: An `int` specifying the maximum number of L-BFGS steps
          per relax iteration. A value of 0 specifies no limit.
      tolerance: kcal/mol, the energy tolerance of L-BFGS.
          The default value is the OpenMM default.
      restraint_set: The set of atoms to restrain.
      max_attempts: The maximum number of minimization attempts per iteration.
      checks: Whether to perform cleaning checks.
      exclude_residues: An optional list of zero-indexed residues to exclude from
          restraints.

    Returns:
      out: A dictionary of output values.
    """

    # `protein.to_pdb` will strip any poorly-defined residues so we need to
    # perform this check before `clean_protein`.
    _check_residues_are_well_defined(prot)
    
    pdb_string = clean_protein(prot, checks=checks)

    exclude_residues = exclude_residues or []
    exclude_residues = set(exclude_residues)
    violations = np.inf
    iteration = 0
    last_violations = 0

    while violations > 0 and iteration < max_outer_iterations:
        ret = _run_one_iteration(
            pdb_string=pdb_string,
            exclude_residues=exclude_residues,
            max_iterations=max_iterations,
            tolerance=tolerance,
            stiffness=stiffness,
            restraint_set=restraint_set,
            max_attempts=max_attempts,
            use_gpu=use_gpu,
        )
        
        headers = protein.get_pdb_headers(prot)    
        if(len(headers) > 0):
            ret["min_pdb"] = '\n'.join(['\n'.join(headers), ret["min_pdb"]])
        
        prot = protein.from_pdb_string(ret["min_pdb"])
        if place_hydrogens_every_iteration:
            pdb_string = clean_protein(prot, checks=True)
        else:
            pdb_string = ret["min_pdb"]
        ret.update(get_violation_metrics(prot))
        ret.update(
            {
                "num_exclusions": len(exclude_residues),
                "iteration": iteration,
            }
        )
        violations = ret["violations_per_residue"]
        exclude_residues = exclude_residues.union(ret["residue_violations"])

        logging.info(
            "Iteration completed: Einit %.2f Efinal %.2f Time %.2f s "
            "num residue violations %d num residue exclusions %d ",
            ret["einit"],
            ret["efinal"],
            ret["opt_time"],
            ret["num_residue_violations"],
            ret["num_exclusions"],
        )
        # TODO: 在relax过程中是否需要添加一个early stopping的条件
        if last_violations == violations: # early stopping
            break

        last_violations = violations
        iteration += 1

    # Check if there are any amide bonds that need to be fixed.
    pdb_string_init = ret["min_pdb"]
    fixer = AmideBondFixer(pdb_string_init)
    pdb_string_checked, info = fixer.process()
    if pdb_string_checked != None:
        pdb_string_hydrated = cleanup.fix_pdb(io.StringIO(pdb_string_checked), {})

        ret = _run_one_iteration(
            pdb_string=pdb_string_hydrated,
            exclude_residues=[],
            max_iterations=max_iterations,
            tolerance=tolerance,
            stiffness=stiffness,
            restraint_set=restraint_set,
            max_attempts=max_attempts,
            use_gpu=use_gpu,
            mobile_residue_indices=info["deleted_residues_ids"],
        )
        headers = protein.get_pdb_headers(prot)
        if(len(headers) > 0):
            ret["min_pdb"] = '\n'.join(['\n'.join(headers), ret["min_pdb"]])
        
        prot = protein.from_pdb_string(ret["min_pdb"])
        if place_hydrogens_every_iteration:
            pdb_string_hydrated = clean_protein(prot, checks=True)
        else:
            pdb_string_hydrated = ret["min_pdb"]
        ret.update(get_violation_metrics(prot))
    
    return ret
