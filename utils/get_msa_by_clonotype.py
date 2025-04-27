import sys
import os
import csv
import itertools
import pickle
import time
import numpy as np
import argparse
import logging
import glob
from time import time
from utils.fasta import (
    read_fasta_file,
    write_fasta_file,
    save_data_to_pickle,
    read_data_from_pickle,
)
from utils.align import run_alignment, get_clonotype
from utils.database import clone_heavy_database_path, clone_light_database_path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from utils.get_chain_info import AntiBody, AntiBodySingle

from utils.multiprocess import dynamic_executor_context

def read_seq_clonotyep(path):
    reader = csv.reader(open(path, "r"))
    header = next(reader)

    # clonotype_data = list(zip(*reader))
    clonotype_data = []
    for i in range(len(header)):
        clonotype_data.append([])
    for row in reader:
        for i in range(len(row)):
            clonotype_data[i].append(row[i])

    return clonotype_data, header


def clone_seq_dict(clonotype_data):
    clones = clonotype_data[0]
    sequences = clonotype_data[2]
    abundacnes = clonotype_data[3]
    result = {}
    for i in range(len(clones)):
        clone = clones[i]
        sequence = sequences[i]
        abundacne = abundacnes[i]
        if clone not in result:
            result[clone] = [[sequence, int(abundacne)]]
        else:
            result[clone].append([sequence, int(abundacne)])
    return result


def cdr3_length_match(cdr3_1, cdr3_2, diff=0):
    if abs(len(cdr3_1) - len(cdr3_2)) > diff:
        return False
    else:
        return True


def gene_similarity(gene_1, gene_2, split="*"):
    index_1 = gene_1.find(split)
    index_2 = gene_2.find(split)
    return gene_1 == gene_2 or gene_1[:index_1] == gene_2[:index_2]


def clonotype_to_dict(clonotype, paired=False):
    clonotype_list = clonotype.split(";")
    result = {}
    if paired:
        result["HV"] = clonotype_list[0].split("|")[0]
        result["HJ"] = clonotype_list[1].split("|")[0]
        result["HCDR3"] = clonotype_list[2]
        result["LV"] = clonotype_list[3].split("|")[0]
        result["LJ"] = clonotype_list[4].split("|")[0]
        result["LCDR3"] = clonotype_list[5]
    else:
        result["V"] = clonotype_list[0].split("|")[0]
        result["J"] = clonotype_list[1].split("|")[0]
        result["CDR3"] = clonotype_list[2]
    return result


def find_matching_group(
    clonotype_dict, target, diff=0, paired=False, v_split="-", j_split="*"
):
    group_list = []
    target_dict = clonotype_to_dict(target, paired)
    for group in clonotype_dict.keys():
        group_dict = clonotype_to_dict(group, paired)
        if paired:
            if (
                gene_similarity(group_dict["HV"], target_dict["HV"], v_split)
                and gene_similarity(group_dict["HJ"], target_dict["HJ"], v_split)
                and cdr3_length_match(
                    group_dict["HCDR3"], target_dict["HCDR3"], diff=diff
                )
                and gene_similarity(group_dict["LV"], target_dict["LV"], v_split)
                and gene_similarity(group_dict["LJ"], target_dict["LJ"], split=j_split)
                and cdr3_length_match(
                    group_dict["LCDR3"], target_dict["LCDR3"], diff=diff
                )
            ):
                group_list.extend(clonotype_dict[group])
        else:
            if (
                gene_similarity(group_dict["V"], target_dict["V"], v_split)
                and gene_similarity(group_dict["J"], target_dict["J"], j_split)
                and cdr3_length_match(
                    group_dict["CDR3"], target_dict["CDR3"], diff=diff
                )
            ):
                group_list.extend(clonotype_dict[group])
    return group_list


def find_matching_group_by_seq(
    target_clonotype,
    target_seq,
    clone_data,
    outfile,
    max_diff=10,
    min_seqs=100,
    paired=False,
):
    clone_dict = clone_seq_dict(clone_data)
    split_char_list = [("*", "*"), ("-", "*"), ("V", "J")]
    return_flag = False
    for index in range(max_diff):
        for split_char in split_char_list:
            group_list = find_matching_group(
                clone_dict,
                target_clonotype,
                index,
                paired,
                split_char[0],
                split_char[1],
            )
            if len(group_list) > min_seqs:
                return_flag = True
                break
        if return_flag:
            break
    seqs_other = [item[0].replace("-", "") for item in group_list]
    seqs = [target_seq] + seqs_other
    names = [">seq_{}".format(index) for index in range(len(seqs))]

    with open(outfile, "w") as f:
        f.write("\n".join(itertools.chain(*zip(names, seqs))))

    return len(names)


def delete_msa_by_first_seq(msa_path):
    """Remove columns from the MSA if the first sequence has a gap in that column."""
    names_list, seqs_list = read_fasta_file(msa_path)
    target_seq = seqs_list[0]
    delete_index_list = [index for index, char in enumerate(target_seq) if char == "-"]

    seqs_list = np.array([list(seq) for seq in seqs_list])
    seqs_list = np.delete(seqs_list, delete_index_list, axis=1)
    seqs_list = seqs_list.tolist()
    seqs_list = ["".join(chars) for chars in seqs_list]

    return names_list, seqs_list


def realign_msa(
    out_fasta_temp_path,
    out_msa_path,
    scheme,
    cpus,
    chain_type="H",
):
    if out_fasta_temp_path is None or out_msa_path is None:
        return
    run_alignment(
        fas_file=out_fasta_temp_path,
        out_path=out_msa_path,
        scheme=scheme,
        chain=chain_type,
        cpus=cpus,
    )
    msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_msa_path)
    write_fasta_file(msa_names_list, msa_seqs_list, out_msa_path)

    if os.path.exists(out_fasta_temp_path):
        os.remove(out_fasta_temp_path)


def search_msa_by_clonotype(
    databases_path,
    antibody,
    tmp_dir,
    output_alignments_dir,
    use_precomputed_msas,
    clone_database_path,
    chain_type="H",
):
    # if chain_type == "H":
    #     antibody = antibody.get_heavy_antibody()

    # else:
    #     antibody = antibody.get_light_antibody()

    if antibody is None:
        return []

    extend_name = False
    if len(clone_database_path.items()) > 1:
        extend_name = True

    result_list = []

    for scheme, database_path in clone_database_path.items():
        feature_dict = {
            "out_fasta_temp_path": None,
            "out_msa_path": None,
            "chain_type" : chain_type,
        }

        name = antibody.name
        database_path = os.path.join(databases_path, database_path)
        msa_out_dir = "{}/{}".format(output_alignments_dir, name)
        if not os.path.exists(msa_out_dir):
            os.makedirs(msa_out_dir, exist_ok=True)
        out_fasta_temp_path = "{}/clonotype_{}_{}_msa_{}.fas".format(
            tmp_dir, scheme, antibody.name, chain_type
        )
        out_msa_path = os.path.join(msa_out_dir, "clonotype_hits")
        if extend_name:
            out_msa_path += "_{}.a3m".format(scheme)
        else:
            out_msa_path += ".a3m"

        if use_precomputed_msas and os.path.exists(out_msa_path):
            result_list.append(feature_dict)
            continue

        clone_data = read_data_from_pickle(database_path)
        find_matching_group_by_seq(
            target_clonotype=antibody.clonotype,
            target_seq=antibody.seq.replace("*", ""),
            clone_data=clone_data,
            outfile=out_fasta_temp_path,
        )
        feature_dict["out_fasta_temp_path"] = out_fasta_temp_path
        feature_dict["out_msa_path"] = out_msa_path

        result_list.append(feature_dict)

    return result_list


def build_msa(args, antibody, chain_type):
    tmp_dir = args.temp_dir
    output_dir = args.output_dir
    databases_path = args.databases_path
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    use_precomputed_msas = args.use_precomputed_msas

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)
    
    results = search_msa_by_clonotype(
        databases_path,
        antibody,
        tmp_dir,
        out_alignments_dir,
        use_precomputed_msas,
        clone_heavy_database_path,
        chain_type=chain_type,
    )
    
    return results


def getmsa(args, antibody_list):
    cpus = args.cpus
    fasta_dir = args.fasta_dir
    tmp_dir = args.temp_dir
    output_dir = args.output_dir
    databases_path = args.databases_path
    scheme = "chothia"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    use_precomputed_msas = args.use_precomputed_msas
    fasta_names = os.listdir(fasta_dir)
    fasta_path = [os.path.join(fasta_dir, name) for name in fasta_names]
    time_start = time()

    print("\n# Build alignments for clonotype ...")

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)

    h_task_count = sum([1 for antibody in antibody_list for _ in antibody.heavy_antibody])
    l_task_count = sum([1 for antibody in antibody_list for _ in antibody.light_antibody])
    
    with  dynamic_executor_context(process_threshold=3, max_workers=cpus) as dynamic_executor:
        executor = dynamic_executor.get_executor(h_task_count + l_task_count)
        futures = []
        for antibody in antibody_list:
            for heavy_chain in antibody.heavy_antibody:
                futures.append(
                    executor.submit(
                        search_msa_by_clonotype,
                        databases_path,
                        heavy_chain,
                        tmp_dir,
                        out_alignments_dir,
                        use_precomputed_msas,
                        clone_heavy_database_path,
                        chain_type="H",
                    )
                )
            for light_chainn in antibody.light_antibody:
                futures.append(
                    executor.submit(
                        search_msa_by_clonotype,
                        databases_path,
                        light_chainn,
                        tmp_dir,
                        out_alignments_dir,
                        use_precomputed_msas,
                        clone_light_database_path,
                        chain_type="L",
                    )
                )

        with ThreadPoolExecutor(max_workers=1) as realign_executor:
            for future in as_completed(futures):
                result_list = future.result()
                for result_dict in result_list:
                    realign_executor.submit(
                        realign_msa,
                        result_dict["out_fasta_temp_path"],
                        result_dict["out_msa_path"],
                        scheme,
                        cpus,
                        chain_type=result_dict["chain_type"],
                    )

    time_end = time()
    print("# Clonotype finished, time cost: {:.2f}s\n".format(time_end - time_start))
