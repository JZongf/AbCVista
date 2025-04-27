from openfold.np.residue_constants import ID_TO_HHBLITS_AA
from itertools import chain
import numpy as np

def save_msa(
    msa: np.ndarray,
    output_file: str,
):
    """
    Save msa to file.
    """
    with open(output_file, "w") as f:
        names = [">seq_{}".format(i) for i in range(msa.shape[0])]
        seqs = ["".join([ID_TO_HHBLITS_AA[aa_id] for aa_id in seq]) for seq in msa]
        f.write("\n".join(chain(*zip(names, seqs))))