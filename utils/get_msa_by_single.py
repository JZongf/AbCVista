import os
from time import time
from typing import Dict
from utils.fasta import read_fasta_file, write_fasta_file
from utils.align import run_alignment, delete_msa_by_first_seq
from utils.database import unpaired_database_path
from utils.get_msa_utils import hamming_distance, split_list
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from utils.get_chain_info import AntiBody, AntiBodySingle
from utils.multiprocess import dynamic_executor_context

class UnpairedFasta:
    def __init__(
        self,
        cdr1_length: int,
        cdr2_length: int,
        path=None,
    ) -> None:

        self.cdr1_length = cdr1_length
        self.cdr2_length = cdr2_length
        self.path = path


def cdrs_length_match(
    database_lengths: Dict[str, UnpairedFasta], query_length, chaintype="H"
):
    """
    Matches CDR lengths in the database with the query sequence's CDR lengths, 
    returns the closest sequences, and excludes those that are exactly the same (in terms of lengths).

    Args:
        database_lengths: CDR lengths in the database, format: {'cdr3_length':['cdr1_length', 'cdr2_length']}
        query_length: CDR lengths of the query sequence, format: ['cdr1_length', 'cdr2_length', 'cdr3_length']
        chaintype: The type of the chain, 'H' or 'L'
    """
    closest = None
    closest_dist_dict = {}

    for d in database_lengths.get(chaintype + "_" + str(query_length[2]), []):
        dist = abs(int(d.cdr1_length) - int(query_length[0])) + abs(
            int(d.cdr2_length) - int(query_length[1])
        )
        closest = [d.cdr1_length, d.cdr2_length, query_length[2]]
        closest = map(str, closest)
        closest = "_".join(closest)
        closest_dist_dict[d.path] = dist
    # Sort the results by distance
    closest_list = sorted(closest_dist_dict.items(), key=lambda x: x[1])
    # Count the total number of sequences found
    seqs_count = sum(
        [int(path.strip(".fasta").split("_")[-1]) for path in closest_dist_dict.keys()]
    )

    # Prevent cases where the query CDR3 is not found in the database.
    database_lengths_list = [
        [int(l.cdr1_length), int(l.cdr2_length), int(c3.split("_")[1]), l.path]
        for c3, file_info in database_lengths.items()
        for l in file_info
        if l.path.startswith(chaintype)
    ]  # Convert the dict to a list of cdrs_length
    if closest_dist_dict == {} or seqs_count < 1000:
        for d in database_lengths_list:
            dist = (
                abs(d[0] - query_length[0])
                + abs(d[1] - query_length[1])
                + abs(d[2] - query_length[2])
            )
            closest_dist_dict[d[3]] = dist

        # Ensure the total number of found sequences is greater than 1000
        closest_list = sorted(closest_dist_dict.items(), key=lambda x: x[1])
        seqs_count = 0
        for index, (path, _) in enumerate(closest_list):
            seqs_count += int(path.strip(".fasta").split("_")[-1])
            if seqs_count > 1000:
                closest_list = closest_list[: index + 1]
                break

    return closest_list


def realign_seqs(
    tmp_name,
    output_name,
    scheme,
    cpus,
    chain_type="H",
):
    if tmp_name is None or output_name is None:
        return
    else:
        # Align the retrieved sequences
        run_alignment(
            fas_file=tmp_name,
            out_path=output_name,
            scheme=scheme,
            chain=chain_type,
            cpus=cpus,
        )
        heavy_names_list, heavy_seqs_list = delete_msa_by_first_seq(output_name)
        write_fasta_file(heavy_names_list, heavy_seqs_list, output_name)

        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def get_msa_by_regions_length_unpaired(
    seq,
    database_lengths,
    output_path,
    hamming_tolerance=0,
    chain_type="H",
    database_path=None,
    minmin_seqs=1000,
    maxmin_seqs=50000,
):
    assert chain_type in ["H", "L"]

    regioned_seq = seq.replace("-", "")
    seq_without_region = regioned_seq.replace("*", "")
    target_lengths = [len(region) for region in regioned_seq.split("*")]

    query_cdr1_length = target_lengths[1]
    query_cdr2_length = target_lengths[3]
    query_cdr3_length = target_lengths[5]
    query_cdr3 = regioned_seq.split("*")[5]

    # Adjust the minimum sequence count upwards according to the CDR3 length.
    if query_cdr3_length > 10:
        minmin_seqs = minmin_seqs * query_cdr3_length // 10 * 4

    closest_lengths = [query_cdr1_length, query_cdr2_length, query_cdr3_length]

    closest_lengths_list = []  # Stores the sequence length(s) that are nearest to the query sequence CDRs.
    closest_lengths_list = cdrs_length_match(
        database_lengths=database_lengths,
        query_length=[query_cdr1_length, query_cdr2_length, query_cdr3_length],
        chaintype=chain_type,
    )

    closest_idx = 0
    target_seqs_list = []
    while len(target_seqs_list) < minmin_seqs:
        if closest_idx >= len(closest_lengths_list):
            break

        closest_lengths = closest_lengths_list[closest_idx][0].split("_")
        target_database = os.path.join(
            database_path, closest_lengths_list[closest_idx][0]
        )

        with open(target_database, "r") as f:
            lines_list = f.read().splitlines()
            target_names_list_temp = lines_list[0::2]
            target_seqs_list_temp = lines_list[1::2]

        if (
            query_cdr3_length == int(closest_lengths[3])
            and len(target_seqs_list_temp) >= minmin_seqs * 10
        ):
            hamming_distance_cutoff = int(target_lengths[5] * hamming_tolerance)
            target_seqs_list.extend(
                [
                    target_seqs_list_temp[index]
                    for index, t_seq in enumerate(target_names_list_temp)
                    if len(t_seq.lstrip(">")) == query_cdr3_length
                    and hamming_distance(t_seq.lstrip(">"), query_cdr3)
                    <= hamming_distance_cutoff
                ]
            )
        else:
            target_seqs_list.extend(target_seqs_list_temp)
            if len(target_seqs_list) > maxmin_seqs:
                target_seqs_list = target_seqs_list[:maxmin_seqs]
                break

        while len(target_seqs_list) > maxmin_seqs:
            hamming_tolerance = hamming_tolerance * 0.9
            hamming_distance_cutoff = int(target_lengths[5] * hamming_tolerance)
            target_seqs_list = [
                target_seqs_list_temp[index]
                for index, t_seq in enumerate(target_names_list_temp)
                if len(t_seq.lstrip(">")) == query_cdr3_length
                and hamming_distance(t_seq.lstrip(">"), query_cdr3)
                <= hamming_distance_cutoff
            ]
            if hamming_tolerance < 0.1:
                break

        closest_idx += 1

    target_seqs_list = [seq_without_region] + target_seqs_list
    names_list = [">seq_{}".format(i) for i in range(len(target_seqs_list))]
    write_fasta_file(names_list, target_seqs_list, output_path)

    return output_path


def get_msa_by_single(
    antibody: AntiBody,
    scheme,
    temp_dir,
    output_alignments_dir,
    use_precomputed_msas,
    databases_path,
    unpaired_database_length_dict,
    chain_type="H",
    hamming_tolerance=0.5 * 2,
):
    if antibody is None:
        return []

    name = antibody.name
    seq = antibody.seq
    extend_name = False
    if len(unpaired_database_path.items()) > 1:
        extend_name = True

    result_list = []
    for scheme, databse_path in unpaired_database_path.items():
        temp_output_path = os.path.join(
            temp_dir, "single_{}_msa_{}.fas".format(name, scheme)
        )
        msa_out_dir = "{}/{}".format(output_alignments_dir, name)
        if not os.path.exists(msa_out_dir):
            os.makedirs(msa_out_dir, exist_ok=True)
        out_msa_path = os.path.join(msa_out_dir, "single_hits")
        if extend_name:
            out_msa_path += "_{}.a3m".format(scheme)
        else:
            out_msa_path += ".a3m"

        if use_precomputed_msas and os.path.exists(out_msa_path):
            result_list.append(
                {
                    "out_fasta_temp_path": None,
                    "out_msa_path": None,
                    "chain_type": chain_type,
                }
            )
            continue

        tmp_name = get_msa_by_regions_length_unpaired(
            seq,
            unpaired_database_length_dict[scheme],
            temp_output_path,
            hamming_tolerance=hamming_tolerance,
            chain_type=chain_type,
            database_path=os.path.join(databases_path, databse_path),
        )

        result_list.append(
            {
                "out_fasta_temp_path": tmp_name,
                "out_msa_path": out_msa_path,
                "chain_type": chain_type,
            }
        )

    return result_list


def build_msa(args, antibody, scheme, chain_type, unpaired_database_length_dict):
    tmp_dir = args.temp_dir
    output_dir = args.output_dir
    use_precomputed_msas = args.use_precomputed_msas
    databases_path = args.databases_path

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)

    result = get_msa_by_single(
        antibody,
        scheme,
        tmp_dir,
        out_alignments_dir,
        use_precomputed_msas,
        databases_path,
        unpaired_database_length_dict,
        chain_type=chain_type,
        hamming_tolerance=0.5 * 2,
    )
    
    return result


def get_database_length_dict(databases_path):
    # Preload the database to avoid duplicate reads.
    unpaired_database_length_dict = {}
    for scheme, databse_path in unpaired_database_path.items():
        unpaired_database_files = [
            file
            for file in os.listdir(os.path.join(databases_path, databse_path))
            if file.endswith(".fasta") or file.endswith(".fas")
        ]
        
        # Get database files list according to database info file
        file_info = []
        if os.path.exists(
            os.path.join(databases_path, databse_path, "database_info.txt")
        ):
            with open(
                os.path.join(databases_path, databse_path, "database_info.txt"), "r"
            ) as f:
                file_info = f.read().splitlines()
        if file_info != []:
            file_info = [
                file for file in file_info if file in set(unpaired_database_files)
            ]
            unpaired_database_files = file_info

        cdr3_length = [
            name.split("_")[0] + "_" + name.split("_")[3].split(".")[0]
            for name in unpaired_database_files
        ]
        cdr1_2_path = [
            [int(name.split("_")[1]), int(name.split("_")[2]), name]
            for name in unpaired_database_files
        ]
        # Use cdr3_length as the key, and cdr1_2_length as the value. If the cdr3_length is the same, append cdr1_2_length to the value.
        temp_length_dict = {}
        for cdr3, cdr1_2 in zip(cdr3_length, cdr1_2_path):
            file_info = UnpairedFasta(cdr1_2[0], cdr1_2[1], cdr1_2[2])
            if cdr3 not in temp_length_dict.keys():
                temp_length_dict[cdr3] = [file_info]
            else:
                temp_length_dict[cdr3].append(file_info)
        unpaired_database_length_dict[scheme] = temp_length_dict

    return unpaired_database_length_dict


def getmsa(args, antibody_list):
    cpus = args.cpus
    fasta_dir = args.fasta_dir
    tmp_dir = args.temp_dir
    output_dir = args.output_dir
    use_precomputed_msas = args.use_precomputed_msas
    databases_path = args.databases_path
    fasta_names = os.listdir(fasta_dir)
    files_path = [os.path.join(fasta_dir, name) for name in fasta_names]
    time_start = time()
    print("\n# Build alignments for single ...")

    if args.use_precomputed_alignments is None:
        out_alignments_dir = os.path.join(output_dir, "alignments")
    else:
        out_alignments_dir = args.use_precomputed_alignments

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir, exist_ok=True)

    # Preload the database to avoid duplicate reads.
    unpaired_database_length_dict = {}
    for scheme, databse_path in unpaired_database_path.items():
        unpaired_database_files = [
            file
            for file in os.listdir(os.path.join(databases_path, databse_path))
            if file.endswith(".fasta") or file.endswith(".fas")
        ]
        
        # Get database files list according to database info file
        file_info = []
        if os.path.exists(
            os.path.join(databases_path, databse_path, "database_info.txt")
        ):
            with open(
                os.path.join(databases_path, databse_path, "database_info.txt"), "r"
            ) as f:
                file_info = f.read().splitlines()
        if file_info != []:
            file_info = [
                file for file in file_info if file in set(unpaired_database_files)
            ]
            unpaired_database_files = file_info

        cdr3_length = [
            name.split("_")[0] + "_" + name.split("_")[3].split(".")[0]
            for name in unpaired_database_files
        ]
        cdr1_2_path = [
            [int(name.split("_")[1]), int(name.split("_")[2]), name]
            for name in unpaired_database_files
        ]
        # Use cdr3_length as the key, and cdr1_2_length as the value. If the cdr3_length is the same, append cdr1_2_length to the value.
        temp_length_dict = {}
        for cdr3, cdr1_2 in zip(cdr3_length, cdr1_2_path):
            file_info = UnpairedFasta(cdr1_2[0], cdr1_2[1], cdr1_2[2])
            if cdr3 not in temp_length_dict.keys():
                temp_length_dict[cdr3] = [file_info]
            else:
                temp_length_dict[cdr3].append(file_info)
        unpaired_database_length_dict[scheme] = temp_length_dict

    h_task_count = sum([1 for antibody in antibody_list for heavy_chain in antibody.get_heavy_antibody()])
    l_task_count = sum([1 for antibody in antibody_list for light_chain in antibody.get_light_antibody()])
    
    with  dynamic_executor_context(process_threshold=3, max_workers=cpus) as dynamic_executor:
        executor = dynamic_executor.get_executor(h_task_count + l_task_count)
        features = []
        for antibody in antibody_list:
            for heavy_chain in antibody.get_heavy_antibody():
                features.append(
                    executor.submit(
                        get_msa_by_single,
                        heavy_chain,
                        scheme,
                        tmp_dir,
                        out_alignments_dir,
                        use_precomputed_msas,
                        databases_path,
                        unpaired_database_length_dict,
                        chain_type="H",
                        hamming_tolerance=0.5 * 2,
                    )
                )
            for light_chain in antibody.get_light_antibody():
                features.append(
                    executor.submit(
                        get_msa_by_single,
                        light_chain,
                        scheme,
                        tmp_dir,
                        out_alignments_dir,
                        use_precomputed_msas,
                        databases_path,
                        unpaired_database_length_dict,
                        chain_type="L",
                        hamming_tolerance=0.4 * 2,
                    )
                )

        with ThreadPoolExecutor(max_workers=1) as realign_executor:
            for future in as_completed(features):
                feature_list = future.result()
                for result_dict in feature_list:
                    realign_executor.submit(
                        realign_seqs,
                        tmp_name=result_dict["out_fasta_temp_path"],
                        output_name=result_dict["out_msa_path"],
                        scheme=scheme,
                        cpus=cpus,
                        chain_type=result_dict["chain_type"],
                    )

    time_end = time()
    print("# Single finished, time cost: {:.2f}s\n".format(time_end - time_start))
