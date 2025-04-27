import os
import sys

sys.path.append("..")
from utils.fasta import read_fasta_file
import numpy as np

Abalign_path = "../lib/Abalign"


def init_Abalign_path(path):
    global Abalign_path

    Abalign_path = os.path.join(path, "Abalign")


def run_alignment(
    fas_file,
    out_path,
    scheme="chothia",
    chain="H",
    get_regioned_file=False,
    merge=False,
    cpus=8,
    cutoff=60,
    align_info=False,
):
    scheme_dict = {"chothia": "-c", "imgt": "-g", "kabat": "-k", "martin": "-m"}
    scheme_cmd = scheme_dict[scheme]
    base_name = os.path.basename(fas_file).split(".")[0]

    cmd = "{} -i {} -s {} {} {} -t {} -z {} -lfs 0".format(
        Abalign_path,
        fas_file,
        scheme_cmd,
        "-ah" if chain == "H" else "-al",
        out_path,
        cpus,
        cutoff,
    )
    if get_regioned_file:
        cmd += " -r "
    if merge:
        cmd += " -mg -bd "
    print(f"alignment {chain} cmd:", cmd)
    info = os.popen(cmd=cmd)
    stream = info.read()
    if align_info:
        print(stream)


def get_clonotype(
    fas_file,
    out_dir,
    scheme="chothia",
    cutoff=60,
    align_info=False,
    chain_type="H",
    regioned_file=False,
    cpus=8,
):
    # Save the original working directory
    original_path = os.getcwd()
    # Get the path of the current file
    current_path = os.path.dirname(os.path.realpath(__file__))
    parenn_path = os.path.dirname(current_path)
    # Change the current working directory, 
    # because Abalign uses the current working directory as the base when searching for the VJ gene library.
    os.chdir(parenn_path)

    scheme_dict = {"chothia": "-c", "imgt": "-g", "kabat": "-k", "martin": "-m"}
    scheme_cmd = scheme_dict[scheme]
    base_name = os.path.basename(fas_file).split(".")[0]
    out_fas_path_heavy = os.path.join(out_dir, "{}_heavy.fas".format(base_name))
    out_fas_path_light = os.path.join(out_dir, "{}_light.fas".format(base_name))
    out_vgene_path_heavy = os.path.join(out_dir, "{}_heavy.vgene".format(base_name))
    out_vgene_path_light = os.path.join(out_dir, "{}_light.vgene".format(base_name))
    if chain_type == "H":
        cmd = "{} -i {} -s {} -ah {} -v {} -vct -sp HS. -z 60 -lfs 0 {} -t {}".format(
            Abalign_path,
            fas_file,
            scheme_cmd,
            out_fas_path_heavy,
            out_vgene_path_heavy,
            "-r" if regioned_file else "",
            cpus,
        )
        print("alignment H cmd: {}".format(cmd))
    elif chain_type == "L":
        cmd = "{} -i {} -s {} -al {} -v {} -vct -sp HS. -z 60 -lfs 0 {} -t {}".format(
            Abalign_path,
            fas_file,
            scheme_cmd,
            out_fas_path_light,
            out_vgene_path_light,
            "-r" if regioned_file else "",
            cpus,
        )
        print("alignment L cmd: {}".format(cmd))
    else:
        raise ValueError("chain_type should belong to ['H', 'L']")

    info = os.popen(cmd=cmd)
    stream = info.read()
    if align_info:
        print(stream)

    # Switch back to the original working directory
    os.chdir(original_path)


def delete_msa_by_first_seq(msa_path):
    """Delete the columns in the MSA that contain gaps in the first sequence"""
    names_list, seqs_list = read_fasta_file(msa_path)
    target_seq = seqs_list[0]
    delete_index_list = [index for index, char in enumerate(target_seq) if char == "-"]

    seqs_list = np.array([list(seq) for seq in seqs_list])
    seqs_list = np.delete(seqs_list, delete_index_list, axis=1)
    seqs_list = seqs_list.tolist()
    seqs_list = ["".join(chars) for chars in seqs_list]

    return names_list, seqs_list
