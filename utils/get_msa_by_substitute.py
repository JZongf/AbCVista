import os
import pandas as pd
import numpy as np
from time import time
from utils.fasta import read_fasta_file, write_fasta_file, read_data_from_pickle
from utils.align import run_alignment, delete_msa_by_first_seq
from utils.database import (
    heavy_length_databases,
    light_length_databases,
    heavy_ab_database_path,
    light_ab_database_path,
)
from utils.get_msa_utils import find_sequences, hamming_distance
from utils.get_chain_info import AntiBody
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp
from utils.multiprocess import dynamic_executor_context
from utils.database import regions

heavy_lengths_sub_list = ([])
heavy_seqs_sub_list = ([])
light_seqs_sub_list = ([])
light_lengths_sub_list = ([])


def realign_seqs(
    tmp_fasta_name,
    out_msa_path,
    scheme,
    cpus,
    chain_type="H",
):
    if tmp_fasta_name is None or out_msa_path is None:
        return
    run_alignment(
        fas_file=tmp_fasta_name,
        out_path=out_msa_path,
        scheme=scheme,
        chain=chain_type,
        cpus=cpus,
    )
    heavy_names_list, heavy_seqs_list = delete_msa_by_first_seq(out_msa_path)
    write_fasta_file(heavy_names_list, heavy_seqs_list, out_msa_path)

    if os.path.exists(tmp_fasta_name):
        os.remove(tmp_fasta_name)


def get_msa_by_regions_length_substitution(
    antibody,
    tmp_dir,
    tolerance=[0],  # Tolerance for the sequence lengths of various regions
    chain_type="H",   # The type of the chain
    database="uniref90",  # Database type
    minmin_seqs=1000,  # Minimum number of sequences (requires 1000 for osa pair)
    maxmin_seqs=10000,  # Maximum number of sequences (requires 50000 for osa pair)
    database_index=0,  # Index of the database being used
    sub_length_max=None,  # Maximum length in the replacement database
    heavy_seqs_sub_list=None,  # Sequences from the replacement heavy chain length database
    heavy_lengths_sub_list=None,  # Lengths from the replacement heavy chain length database
    light_seqs_sub_list=None,  # Sequences from the replacement light chain length database
    light_lengths_sub_list=None,  # Lengths from the replacement light chain length database
):
    assert chain_type in ["H", "L"]
    name = antibody.name
    seq = antibody.seq

    target_names = []
    cycle = 0

    output_path = os.path.join(tmp_dir, "{}_{}_msa.fas".format(database, name))

    regioned_seq = seq.replace("-", "")
    target_lengths = [len(region) for region in regioned_seq.split("*")]
    seq_without_region = regioned_seq.replace("*", "")
    # A tolerance of 0.1 is allowed by default in the CDR3 region. It might be necessary to check if tolerance should be applied to the light chain.
    # tolerance[5] += int(target_lengths[5] * 0.1) 

    upper_count_list = [
        0 for _ in range(len(target_lengths))
    ]  # Used to store the count of regions whose lengths exceed the maximum length in the database.
    if sub_length_max is not None:
        for i in range(len(target_lengths)):
            if target_lengths[i] > sub_length_max[i]:
                region_diff = target_lengths[i] - sub_length_max[i]
                upper_count_list[i] = region_diff
        target_lengths = [
            min(target_lengths[i], sub_length_max[i])
            for i in range(len(target_lengths))
        ]

    if chain_type == "H":
        target_lengths_df = np.tile(target_lengths, (heavy_seqs_sub_list[database_index].shape[0], 1))
    else:
        target_lengths_df = np.tile(target_lengths, (light_seqs_sub_list[database_index].shape[0], 1))
        
    target_df = pd.DataFrame(target_lengths_df, columns=regions)
    
    while len(target_names) < minmin_seqs and cycle < 50:
        if chain_type == "H":
            target_seqs = find_sequences(
                seqs_df=heavy_seqs_sub_list[database_index],
                length_df=heavy_lengths_sub_list[database_index],
                target_df=target_df,
                tolerance=tolerance,
            )
        else:
            target_seqs = find_sequences(
                seqs_df=light_seqs_sub_list[database_index],
                length_df=light_lengths_sub_list[database_index],
                target_df=target_df,
                tolerance=tolerance,
            )

        target_seqs = target_seqs.to_numpy().tolist()
        if len(target_seqs) >= minmin_seqs or cycle == 49:
            if len(target_seqs) > maxmin_seqs:
                target_seqs = [
                    seq
                    for seq in target_seqs
                    if len(regioned_seq.split("*")[5]) == len(seq[5])
                    and hamming_distance(regioned_seq.split("*")[5], seq[5])
                    <= int(target_lengths[5] * 0.8 * 2)
                ]
            target_seqs = target_seqs[:maxmin_seqs]
            target_seqs = ["".join(seq) for seq in target_seqs]
            target_seqs = [seq_without_region] + target_seqs
            target_names = [">seq_{}".format(i) for i in range(len(target_seqs))]
            write_fasta_file(target_names, target_seqs, output_path)
            break
        else:
            if sum(upper_count_list) > 0:
                for i, count in enumerate(upper_count_list):
                    if count > 0:
                        tolerance[i] = tolerance[i] + 1
                        upper_count_list[i] = count - 1
                        break
            else:
                tolerance[cycle % 7] = tolerance[cycle % 7] + 1
            cycle += 1

    return output_path


def search_msas_by_substitution(
    databases_path,
    sub_database_path,
    tmp_dir,
    out_alignments_dir,
    antibody,
    use_precomputed_msas,
    sub_length_max,
    chain_type="H",
):

    if antibody is None:
        return []

    extend_name = False
    if len(sub_database_path) > 1:
        extend_name = True

    result_list = []
    for index, (database_name, database_path) in enumerate(sub_database_path.items()):
        # 生成重链的MSA
        result_dict = {
            "out_fasta_temp_path": None,
            "out_msa_path": None,
            "chain_type": chain_type,
        }

        out_msa_dir = os.path.join(out_alignments_dir, antibody.name)
        if not os.path.exists(out_msa_dir):
            os.makedirs(out_msa_dir, exist_ok=True)

        out_msa_path = os.path.join(
            out_msa_dir,
            "uniref90_hits",
        )
        if extend_name:
            out_msa_path += "_" + database_name + ".a3m"

        else:
            out_msa_path += ".a3m"

        if use_precomputed_msas and os.path.exists(out_msa_path):
            result_list.append(result_dict)
            continue

        seqs_sub_list = []
        lengths_sub_list = []
        database_full_path = os.path.join(databases_path, database_path)
        seqs, lengths = read_data_from_pickle(database_full_path)
        seqs_sub_list.append(seqs)
        lengths_sub_list.append(lengths)

        if chain_type == "H":
            tmp_fasta_name = get_msa_by_regions_length_substitution(
                antibody=antibody,
                tmp_dir=tmp_dir,
                tolerance=[10, 0, 0, 0, 0, 0, 5],
                chain_type=chain_type,
                database="uniref90",
                database_index=index,
                sub_length_max=sub_length_max,
                heavy_lengths_sub_list=lengths_sub_list,
                heavy_seqs_sub_list=seqs_sub_list,
            )
        else:
            tmp_fasta_name = get_msa_by_regions_length_substitution(
                antibody=antibody,
                tmp_dir=tmp_dir,
                tolerance=[10, 0, 10, 0, 10, 0, 10],
                chain_type=chain_type,
                database="uniref90",
                database_index=index,
                sub_length_max=sub_length_max,
                light_lengths_sub_list=lengths_sub_list,
                light_seqs_sub_list=seqs_sub_list,
            )

        result_dict["out_fasta_temp_path"] = tmp_fasta_name
        result_dict["out_msa_path"] = out_msa_path
        result_list.append(result_dict)

    return result_list


def build_msa(args, antibody, scheme, chain_type, database_path, sub_length_max):
    fasta_dir = args.fasta_dir
    tmp_dir = args.temp_dir
    output_dir = args.output_dir

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)
        
    result = search_msas_by_substitution(
        args.databases_path,
        database_path,
        tmp_dir,
        out_alignments_dir,
        antibody,
        args.use_precomputed_msas,
        sub_length_max,
        chain_type=chain_type,
    )
    
    return result


def get_sub_max_length(databases_path, database_heavy, database_light):
    # Preload the sequences and lengths from the replacement database to avoid duplicate file reads.
    global heavy_seqs_sub_list, heavy_lengths_sub_list, light_seqs_sub_list, light_lengths_sub_list
    for _, database_path in database_heavy.items():
        database_path = os.path.join(databases_path, database_path)
        heavy_seqs, heavy_lengths = read_data_from_pickle(database_path)
        heavy_seqs_sub_list.append(heavy_seqs)
        heavy_lengths_sub_list.append(heavy_lengths)

    for _, database_path in database_light.items():
        database_path = os.path.join(databases_path, database_path)
        light_seqs, light_lengths = read_data_from_pickle(database_path)
        light_seqs_sub_list.append(light_seqs)
        light_lengths_sub_list.append(light_lengths)

    sub_length_heavy_max = (
        heavy_lengths_sub_list[0].max(axis=0).tolist()
    )
    sub_length_light_max = (
        light_lengths_sub_list[0].max(axis=0).tolist()
    )

    return sub_length_heavy_max, sub_length_light_max


def getmsa(args, antibody_list):
    cpus = args.cpus
    fasta_dir = args.fasta_dir
    tmp_dir = args.temp_dir
    output_dir = args.output_dir
    use_precomputed_msas = args.use_precomputed_msas
    databases_path = args.databases_path
    fasta_names = os.listdir(fasta_dir)
    files_path = [os.path.join(fasta_dir, name) for name in fasta_names]
    database_name = "uniref90"
    scheme = "chothia"
    time_start = time()
    print("\n# Build alignments for sub ...")

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)

    global heavy_seqs_sub_list, heavy_lengths_sub_list, light_seqs_sub_list, light_lengths_sub_list
    for _, database_path in heavy_length_databases.items():
        database_path = os.path.join(databases_path, database_path)
        heavy_seqs, heavy_lengths = read_data_from_pickle(database_path)
        heavy_seqs_sub_list.append(heavy_seqs)
        heavy_lengths_sub_list.append(heavy_lengths)

    for _, database_path in light_length_databases.items():
        database_path = os.path.join(databases_path, database_path)
        light_seqs, light_lengths = read_data_from_pickle(database_path)
        light_seqs_sub_list.append(light_seqs)
        light_lengths_sub_list.append(light_lengths)

    sub_length_heavy_max = (
        heavy_lengths_sub_list[0].max(axis=0).tolist()
    )
    sub_length_light_max = (
        light_lengths_sub_list[0].max(axis=0).tolist()
    )


    h_task_count = sum([1 for antibody in antibody_list for heavy_chain in antibody.get_heavy_antibody()])
    l_task_count = sum([1 for antibody in antibody_list for light_chain in antibody.get_light_antibody()])

    with  dynamic_executor_context(process_threshold=3, max_workers=cpus) as dynamic_executor:
        executor = dynamic_executor.get_executor(h_task_count + l_task_count)
        futures = []
        for antibody in antibody_list:
            for heavy_chain in antibody.get_heavy_antibody():
                futures.append(
                    executor.submit(
                        search_msas_by_substitution,
                        databases_path=databases_path,
                        sub_database_path=heavy_length_databases,
                        tmp_dir=tmp_dir,
                        out_alignments_dir=out_alignments_dir,
                        antibody=heavy_chain,
                        use_precomputed_msas=use_precomputed_msas,
                        sub_length_max=sub_length_heavy_max,
                        chain_type="H",
                    )
                )
            for light_chain in antibody.get_light_antibody():
                futures.append(
                    executor.submit(
                        search_msas_by_substitution,
                        databases_path=databases_path,
                        sub_database_path=light_length_databases,
                        tmp_dir=tmp_dir,
                        out_alignments_dir=out_alignments_dir,
                        antibody=light_chain,
                        use_precomputed_msas=use_precomputed_msas,
                        sub_length_max=sub_length_light_max,
                        chain_type="L",
                    )
                )

        with ThreadPoolExecutor(max_workers=1) as realign_executor:
            for future in as_completed(futures):
                future_list = future.result()
                for result_dict in future_list:
                    realign_executor.submit(
                        realign_seqs,
                        result_dict["out_fasta_temp_path"],
                        result_dict["out_msa_path"],
                        scheme,
                        cpus,
                        chain_type=result_dict["chain_type"],
                    )


    time_end = time()
    print("# Sub finished, time cost: {:.2f}s\n".format(time_end - time_start))
