import os
import numpy as np
import pandas as pd
from time import time
from utils.fasta import (
    read_fasta_file,
    write_fasta_file,
    read_data_from_pickle,
    merge_fasta_file,
)
from utils.align import run_alignment, delete_msa_by_first_seq
from utils.database import fv_length_database_path
from utils.get_msa_utils import find_sequences, hamming_distance
from utils.get_chain_info import AntiBody, PairAntiBody
from utils.database import regions, regions_fv
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
import random
from utils.multiprocess import dynamic_executor_context

fv_seqs = None  # Global variable to store Fv region sequences, avoiding duplicate file reads.
fv_lengths = None  # Global variable to store the Fv region length, avoiding duplicate file reads.


def realign_seqs(
    out_temp_name_list,
    out_name_list,
    cpus,
    pair_idx=None,
    idx_start=0,
):
    run_alignment(
        fas_file=out_temp_name_list[0],
        out_path=out_name_list[0],
        scheme="chothia",
        chain="H",
        cpus=cpus,
    )
    msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[0])
    
    A_id_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    B_id_list = ['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    if isinstance(pair_idx, int):
        msa_names_list = [
            ">tr|A0A8T2NDK6|A0A8T2NDK6_{}/{}{}-{}{}".format(
                index+idx_start,
                A_id_list[pair_idx%len(A_id_list)], 
                index+idx_start,
                B_id_list[pair_idx%len(B_id_list)], 
                index+idx_start,
            )
            for index, name in enumerate(msa_names_list, 1)
        ]
    else:
        msa_names_list = [
            ">tr|A0A8T2NDK6|A0A8T2NDK6_{}/H{}-L{}".format(index, index, index)
            for index, name in enumerate(msa_names_list, 1)
        ]
    
    msa_names_list[0] = ">Heavy_chain"
    write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[0])

    run_alignment(
        fas_file=out_temp_name_list[1],
        out_path=out_name_list[1],
        scheme="chothia",
        chain="L",
        cpus=cpus,
    )
    msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[1])
    if isinstance(pair_idx, int):
        msa_names_list = [
            ">tr|A0A8T2NDK6|A0A8T2NDK6_{}/{}{}-{}{}".format(
                index+idx_start, 
                B_id_list[pair_idx%len(A_id_list)], 
                index+idx_start, 
                A_id_list[pair_idx%len(B_id_list)], 
                index+idx_start,
            )
            for index, name in enumerate(msa_names_list, 1)
        ]
    else:
        msa_names_list = [
            ">tr|A0A8T2NDK6|A0A8T2NDK6_{}/L{}-H{}".format(index, index, index)
            for index, name in enumerate(msa_names_list, 1)
        ]
        
    msa_names_list[0] = ">Light_chain"
    write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[1])

    for file_name in out_temp_name_list:
        if os.path.exists(file_name):
            os.remove(file_name)

    return len(msa_names_list)

def process_feature(feature_list, cpus, idx_start=0):
    for feature in feature_list:
        if feature["out_temp_name_list"] is None or feature["out_name_list"] is None:
            continue
        idx_start += realign_seqs(
            feature["out_temp_name_list"],
            feature["out_name_list"],
            cpus,
            pair_idx=feature.get("pair_idx", None),
            idx_start=idx_start,
        )
    return idx_start

def get_msa_by_regions_length_paired(
    heavy_antibody,
    light_antibody,
    tmp_dir,
    tolerance=[],
    minmin_seqs=2000,  # Minimum number of sequences
    minmax_seqs=50000,  # Maximum number of sequences
    fv_lengths_max=None,  # Maximum length of the Fv region
    fv_lengths_qx=None,  # Upper x-quantile of the Fv region length
    fv_seqs=None,  # Fv region sequences
    fv_lengths=None,  # Fv region lengths
    scheme="chothia",
):
    target_names = []
    cycle = 0

    seq = heavy_antibody.seq + "*" + light_antibody.seq
    temp_output_heavy = os.path.join(
        tmp_dir, "pair_{}_msa_{}_{}.fas".format(heavy_antibody.name, scheme, random.randint(1,10000))
    )
    temp_output_light = os.path.join(
        tmp_dir, "pair_{}_msa_{}_{}.fas".format(light_antibody.name, scheme, random.randint(1,10000))
    )
    regioned_seq = seq.replace("-", "")
    target_lengths = [len(region) for region in regioned_seq.split("*")]
    # tolerance[5] += int(target_lengths[5] * 0.1) # Default tolerance of 0.1 is allowed

    # Prevent query sequence region lengths from exceeding the maximum length in the database.
    upper_count_list = [
        0 for _ in range(len(target_lengths))
    ]  # Stores the count of regions exceeding the database maximum length, to prioritize processing them.
    if fv_lengths_max is not None:
        for i in range(len(target_lengths)):
            if target_lengths[i] > fv_lengths_max[i]:
                region_diff = target_lengths[i] - fv_lengths_qx[i]
                upper_count_list[i] = region_diff
        target_lengths = [
            min(target_lengths[i], fv_lengths_max[i])
            for i in range(len(target_lengths))
        ]

    target_lengths_df = np.tile(target_lengths, (fv_seqs.shape[0], 1))
    target_df = pd.DataFrame(target_lengths_df, columns=regions_fv)

    while len(target_names) < minmin_seqs and cycle < 100:
        target_seqs = find_sequences(
            seqs_df=fv_seqs,
            length_df=fv_lengths,
            target_df=target_df,
            tolerance=tolerance,
        )
        target_seqs = target_seqs.to_numpy().tolist()

        if len(target_seqs) >= minmin_seqs or cycle == 99:
            if len(target_seqs) > minmax_seqs:
                target_seqs = [
                    seq
                    for seq in target_seqs
                    if len(regioned_seq.split("*")[5]) == len(seq[5])
                    and hamming_distance(regioned_seq.split("*")[5], seq[5])
                    <= int(target_lengths[5] * 0.8 * 2)
                ]
            target_seqs_heavy = ["".join(seq[:7]) for seq in target_seqs]
            target_seqs_light = ["".join(seq[7:]) for seq in target_seqs]

            target_seqs_heavy = [
                "".join(regioned_seq.split("*")[:7])
            ] + target_seqs_heavy
            target_seqs_light = [
                "".join(regioned_seq.split("*")[7:])
            ] + target_seqs_light
            target_names = [">seq_{}".format(i) for i in range(len(target_seqs_heavy))]

            write_fasta_file(target_names, target_seqs_heavy, temp_output_heavy)
            write_fasta_file(target_names, target_seqs_light, temp_output_light)

            break

        else:
            if sum(upper_count_list) > 0:
                for i, count in enumerate(upper_count_list):
                    if count > 0:
                        tolerance[i] = tolerance[i] + 1
                        upper_count_list[i] = count - 1
                        break
            else:
                tolerance[cycle % 14] = tolerance[cycle % 14] + 1
            cycle += 1

    return [temp_output_heavy, temp_output_light]


def search_msas_by_pair(
    antibody,
    tmp_dir,
    out_alignments_dir,
    use_precomputed_msas,
    fv_lengths_max,
    fv_lengths_q3,
    fv_database_path,
    label = "paired_hits",
    pair_idx = None,
):
    # Generate MSA for the paired database
    if pair_idx != None:
        label = label + "_" + str(pair_idx)
    result_list = []
    if antibody.is_paired():
        extend_name = False
        if len(fv_length_database_path.items()) > 1:
            extend_name = True

        for scheme, database_path in fv_length_database_path.items():
            result_dict = {
                "out_temp_name_list": None,
                "out_name_list": None,
            }
            heavy_antibody = antibody.get_heavy_antibody()
            light_antibody = antibody.get_light_antibody()

            heavy_output_path = os.path.join(
                out_alignments_dir, heavy_antibody.name, label
            )
            light_output_path = os.path.join(
                out_alignments_dir, light_antibody.name, label
            )

            if not os.path.exists(os.path.dirname(heavy_output_path)):
                os.makedirs(os.path.dirname(heavy_output_path), exist_ok=True)
            if not os.path.exists(os.path.dirname(light_output_path)):
                os.makedirs(os.path.dirname(light_output_path), exist_ok=True)

            if extend_name:
                heavy_output_path += "_" + scheme + ".a3m"
                light_output_path += "_" + scheme + ".a3m"

            else:
                heavy_output_path += ".a3m"
                light_output_path += ".a3m"

            result_dict["out_name_list"] = [heavy_output_path, light_output_path]
            if use_precomputed_msas:
                if os.path.exists(heavy_output_path) and os.path.exists(
                    light_output_path
                ):
                    result_dict["out_temp_name_list"] = None
                    result_list.append(result_dict)
                    continue

            fv_seqs, fv_lengths = read_data_from_pickle(fv_database_path)
            out_temp_name_list = get_msa_by_regions_length_paired(
                heavy_antibody,
                light_antibody,
                tmp_dir,
                tolerance=[16, 0, 2, 0, 5, 0, 10, 16, 0, 2, 0, 2, 0, 10],
                fv_lengths_max=fv_lengths_max,
                fv_lengths_qx=fv_lengths_q3,
                fv_seqs=fv_seqs,
                fv_lengths=fv_lengths,
                scheme=scheme,
            )
            result_dict["out_temp_name_list"] = out_temp_name_list
            result_dict["pair_idx"] = pair_idx
            result_list.append(result_dict)

    else:
        return []

    return result_list


def build_msa(args, antibody, fv_lengths_max, fv_lengths_q3, fv_database_path, label="paired_hits", pair_idx=None):
    tmp_dir = args.temp_dir
    output_dir = args.output_dir
    use_precomputed_msas = args.use_precomputed_msas

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)
    
    result = search_msas_by_pair(
        antibody,
        tmp_dir,
        out_alignments_dir,
        use_precomputed_msas,
        fv_lengths_max,
        fv_lengths_q3,
        fv_database_path,
        label=label,
        pair_idx=pair_idx,
    )
    
    return result


def get_database_stats(databases_path):
    fv_database_path = os.path.join(databases_path, fv_length_database_path["paired"])
    fv_seqs, fv_lengths = read_data_from_pickle(fv_database_path)
    # Get the maximum value of each column in fv_length
    fv_lengths_max = fv_lengths.max(axis=0).tolist()
    # Get the upper quartile of each column in fv_length
    fv_lengths_q3 = fv_lengths.quantile(0.999).tolist()

    return fv_lengths_max, fv_lengths_q3, fv_database_path


def getmsa(args, antibody_list):
    cpus = args.cpus
    fasta_dir = args.fasta_dir
    tmp_dir = args.temp_dir
    output_dir = args.output_dir
    use_precomputed_msas = args.use_precomputed_msas
    databases_path = args.databases_path
    fasta_names = os.listdir(fasta_dir)
    fasta_path = [os.path.join(fasta_dir, name) for name in fasta_names]
    time_start = time()
    print("\n# Build alignments for pair ...")

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)

    # Preload the database to avoid duplicate reads.
    global fv_seqs, fv_lengths
    fv_database_path = os.path.join(databases_path, fv_length_database_path["paired"])
    fv_seqs, fv_lengths = read_data_from_pickle(fv_database_path)
    # Get the maximum value of each column in fv_length
    fv_lengths_max = fv_lengths.max(axis=0).tolist()
    # Get the upper quartile of each column in fv_length
    fv_lengths_q3 = fv_lengths.quantile(0.999).tolist()

    
    task_count = sum([2 for antibody in antibody_list for pair in antibody.get_all_antibodies() if pair != None])

    with  dynamic_executor_context(process_threshold=3, max_workers=cpus) as dynamic_executor:
        executor = dynamic_executor.get_executor(task_count)

        features = []
        idx_start = 0
        for antibody in antibody_list:
            paired_antibodies = antibody.get_all_antibodies()
            if paired_antibodies == None:
                continue
            if len(paired_antibodies) == 1:
                paired_idx = None
            else:
                paired_idx = 0
            
            for pair in paired_antibodies:
                if isinstance(pair, PairAntiBody):
                    features.append(
                        executor.submit(
                            search_msas_by_pair,
                            pair,
                            tmp_dir,
                            out_alignments_dir,
                            use_precomputed_msas,
                            fv_lengths_max,
                            fv_lengths_q3,
                            fv_database_path,
                            "paired_hits",
                            paired_idx,
                        )
                    )
                    if paired_idx != None:
                        paired_idx += 1

        with ThreadPoolExecutor(max_workers=1) as realign_executor:
            for feature in as_completed(features):
                feature_result = feature.result()
                idx_start += realign_executor.submit(process_feature, feature_result, cpus, idx_start).result()
                if idx_start > 100000000:
                    idx_start = 0
                

    with ProcessPoolExecutor(
        max_workers=cpus, mp_context=mp.get_context("spawn")
    ) as executor:
        features = [
            executor.submit(
                search_msas_by_pair,
                inner_pair,
                tmp_dir,
                out_alignments_dir,
                use_precomputed_msas,
                fv_lengths_max,
                fv_lengths_q3,
                fv_database_path,
                f"inner_pair_hits_{i}",
            )
            for antibody in antibody_list for i, inner_pair in enumerate(antibody.inner_pair_antibody) if inner_pair != None
        ]

        with ThreadPoolExecutor(max_workers=1) as realign_executor:
            for feature in as_completed(features):
                feature_result = feature.result()
                realign_executor.submit(process_feature, feature_result, cpus)
    
    time_end = time()
    print("# Pair finished, time cost: {:.2f}s\n".format(time_end - time_start))
