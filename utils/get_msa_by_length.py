import os
import shutil
import pandas as pd
import csv
from time import time
from typing import Dict
from utils.fasta import read_fasta_file, write_fasta_file, save_data_to_pickle, read_data_from_pickle, merge_fasta_file
from utils.align import run_alignment, delete_msa_by_first_seq
from utils.database import regions, schemes, heavy_length_databases, light_length_databases, \
unpaired_database_path, fv_length_database_path, regions_fv, heavy_ab_database_path, light_ab_database_path

fv_seqs = None # Global variable to store Fv region sequences, avoiding duplicate file reads.
fv_lengths = None # Global variable to store the Fv region length, avoiding duplicate file reads.
heavy_seqs_sub_list = [] # Global variable to store sequences from the replacement light chain length database, avoiding duplicate file reads.
heavy_lengths_sub_list = [] # Global variable to store lengths from the replacement heavy chain length database, avoiding duplicate file reads.
light_seqs_sub_list = [] # Global variable to store sequences from the replacement light chain length database, avoiding duplicate file reads.
light_lengths_sub_list = []  # Global variable to store lengths from the replacement light chain length database, avoiding duplicate file reads.


class UnpairedFasta:
    def __init__(
        self,
        cdr1_length: int,
        cdr2_length: int,
        path = None,
    ) -> None:
        
        self.cdr1_length = cdr1_length
        self.cdr2_length = cdr2_length
        self.path = path        


def get_seqs_length(file_path):
    names_list, seqs_list = read_fasta_file(file_path)
    seqs_list = [seq.replace("-", "").split("*") for seq in seqs_list]
    length_list = [list(map(len, seq)) for seq in seqs_list]
    return length_list, seqs_list


def find_sequences(
    seqs_df, 
    length_df, 
    target_lengths, 
    tolerance=0, 
    type=""
    ):
    target_lengths_df = [target_lengths for _ in range(seqs_df.shape[0])]
    if type == "fv":
        target_df = pd.DataFrame(target_lengths_df, columns=regions_fv)
    else:
        target_df = pd.DataFrame(target_lengths_df, columns=regions)
    
    # Calculate the absolute difference between the target length and each row of the sequence data.
    diff = (length_df - target_df).abs()

    # Filter out rows where the difference is less than or equal to the tolerance.
    matched_sequences = seqs_df[(diff <= tolerance).all(axis=1)]
    
    return matched_sequences


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings"""
    if len(s1) != len(s2):
        raise ValueError("The two strings have different lengths, cannot calculate Hamming distance.")

    distance = 0
    for i in range(len(s1)):
        distance += bin(ord(s1[i]) ^ ord(s2[i])).count('1')
    return distance


def cdrs_length_match(
    database_lengths:Dict[str, UnpairedFasta],
    query_length, 
    chaintype="H"
    ):
    """
    Match the CDR length in the database with the query sequence's CDR length, and return the closest sequence.
    args:
        database_lengths: CDR lengths in the database, format: {'cdr3_length': ['cdr1_length', 'cdr2_length']}
        query_length: CDR lengths of the query sequence, format: ['cdr1_length', 'cdr2_length', 'cdr3_length']
        chaintype: The type of the chain, 'H' or 'L'
    """
    closest = None
    closest_dist_dict = {}
    
    for d in database_lengths.get(chaintype + "_" + str(query_length[2]), []):
        dist = abs(int(d.cdr1_length) - int(query_length[0])) + abs(int(d.cdr2_length) - int(query_length[1]))
        closest = [d.cdr1_length, d.cdr2_length, query_length[2]]
        closest = map(str, closest)
        closest = "_".join(closest)
        closest_dist_dict[d.path] = dist
    # Sort the results by distance
    closest_list = sorted(closest_dist_dict.items(), key=lambda x: x[1])
    # Count the total number of sequences found
    seqs_count = sum([int(path.strip(".fasta").split("_")[-1]) for path in closest_dist_dict.keys()])
    
    # Prevent cases where the query CDR3 is not found in the database.
    database_lengths_list = [[int(l.cdr1_length), int(l.cdr2_length), int(c3.split("_")[1]), l.path] 
                        for c3, file_info in database_lengths.items() for l in file_info if l.path.startswith(chaintype)] # 将dict转换为cdrs_length的list
    if closest_dist_dict == {} or seqs_count < 1000:
        for d in database_lengths_list:
            dist = abs(d[0] - query_length[0]) + abs(d[1] - query_length[1]) +  abs(d[2] - query_length[2])
            closest_dist_dict[d[3]] = dist
        
        # Make sure the total number of found sequences is over 1000
        closest_list = sorted(closest_dist_dict.items(), key=lambda x: x[1])
        seqs_count = 0
        for index, (path, _) in enumerate(closest_list):
            seqs_count += int(path.strip(".fasta").split("_")[-1])
            if seqs_count > 1000:
                closest_list = closest_list[:index+1]
                break
                
    return closest_list


def save_seq_length_to_pickle(seqs_path, output_path):
    length_list, seqs_list = get_seqs_length(seqs_path)
    seqs_df = pd.DataFrame(seqs_list, columns=regions_fv)
    length_df = pd.DataFrame(length_list, columns=regions_fv)
    save_data_to_pickle((seqs_df, length_df), output_path)


def get_msa_by_regions_length_paired(
    file_path, 
    output_dir, 
    tolerance=[],
    chain_type="H", 
    use_precomputed_msas=False, 
    database_path=None, 
    fold_type="alphafold", 
    chain_id=["A", "B"],
    minmin_seqs=2000,
    minmax_seqs=50000,
    fv_lengths_max=None,
    fv_lengths_qx=None,
):
    """
    Finds sequences in the paired sequence database that meet the specified length requirements.

    Args:
        file_path: Path to the input FASTA file.
        output_dir: Path to the output directory.
        tolerance: Tolerance for the sequence lengths of various regions.
        chain_type: The type of the chain.
        use_precomputed_msas: Whether to skip sequences that have already been computed.
        database_path: Path to the database.
        fold_type: The type of fold used for docking.
        chain_id: AlphaFold's chain ID.
        minmin_seqs: Minimum number of sequences.
        minmax_seqs: Maximum number of sequences.
        fv_lengths_max: Maximum length of the Fv region.
        fv_lengths_qx: Upper x-quantile of the Fv region length.
    """
    names_list, seqs_list = read_fasta_file(file_path)
    base_name = os.path.basename(file_path).split(".")[0].rstrip("_merged")
    tolerance_changed_list = [] 
    out_name_list = []
    
    for index, seq in enumerate(seqs_list):
        target_names = []
        tolerance_changed_flag = True
        cycle = 0
        if fold_type == "openfold":
            temp_output_1 = os.path.join(output_dir, names_list[index].split("|")[0].lstrip(">"))
            temp_output_2 = os.path.join(output_dir, names_list[index].split("|")[1].lstrip(">"))
            out_path_1 = os.path.join(temp_output_1, "paired_hits.a3m.tmp")
            out_path_2 = os.path.join(temp_output_2, "paired_hits.a3m.tmp")
        elif fold_type == "alphafold":
            temp_output_1 = "{}/{}/msas/{}".format(output_dir, base_name, chain_id[index][0])
            temp_output_2 = "{}/{}/msas/{}".format(output_dir, base_name, chain_id[index][1])
            out_path_1 = os.path.join(temp_output_1, "paired_hits.a3m.tmp")
            out_path_2 = os.path.join(temp_output_2, "paired_hits.a3m.tmp")
        else:
            raise ValueError("foldtype must be openfold or alphafold")
        
        if not os.path.exists(temp_output_1):
            os.makedirs(temp_output_1)
        if not os.path.exists(temp_output_2):
            os.makedirs(temp_output_2)
        
        if use_precomputed_msas:
            if os.path.exists(out_path_1.rstrip("tmp").rstrip(".")) and os.path.exists(out_path_2.rstrip("tmp").rstrip(".")):
                tolerance_changed_list.append(False)
                out_name_list.append(names_list[index])
                continue
        
        regioned_seq = seq.replace("-", "")
        target_lengths = [len(region) for region in regioned_seq.split("*")]
        seq_without_region = regioned_seq.replace("*", "")
        # tolerance[5] += int(target_lengths[5] * 0.1)
        
        # Prevent query sequence region lengths from exceeding the maximum length in the database.
        # Used to store the count of regions whose lengths exceed the maximum length in the database; 
        # prioritize handling regions that exceed the limit.
        upper_count_list = [0 for _ in range(len(target_lengths))]
        if fv_lengths_max is not None:
            for i in range(len(target_lengths)):
                if target_lengths[i] > fv_lengths_max[i]:
                    region_diff = target_lengths[i] - fv_lengths_qx[i]
                    upper_count_list[i] = region_diff
            target_lengths = [min(target_lengths[i], fv_lengths_max[i]) for i in range(len(target_lengths))]
        
        while len(target_names) < minmin_seqs and cycle < 100:
            target_seqs = find_sequences(seqs_df=fv_seqs, length_df=fv_lengths, target_lengths=target_lengths, tolerance=tolerance, type="fv")
            target_seqs = target_seqs.to_numpy().tolist()
            
            if len(target_seqs) >= minmin_seqs or cycle == 99:
                if len(target_seqs) > minmax_seqs:
                    target_seqs = [seq for seq in target_seqs if len(regioned_seq.split("*")[5]) == len(seq[5]) 
                                   and hamming_distance(regioned_seq.split("*")[5], seq[5]) <= int(target_lengths[5] * 0.8 * 2)]
                target_seqs_heavy = ["".join(seq[:7]) for seq in target_seqs]
                target_seqs_light = ["".join(seq[7:]) for seq in target_seqs]

                target_seqs_heavy = ["".join(regioned_seq.split("*")[:7])] + target_seqs_heavy
                target_seqs_light = ["".join(regioned_seq.split("*")[7:])] + target_seqs_light
                target_names = [">seq_{}".format(i) for i in range(len(target_seqs_heavy))]
                
                write_fasta_file(target_names, target_seqs_heavy, out_path_1)
                write_fasta_file(target_names, target_seqs_light, out_path_2)
                tolerance_changed_list.append(tolerance_changed_flag)
                out_name_list.append([out_path_1, out_path_2])
                break
            
            else:
                if sum(upper_count_list) > 0:
                    for i, count in enumerate(upper_count_list):
                        if count > 0:
                            tolerance[i] = tolerance[i] + 1
                            upper_count_list[i] = count - 1
                            break
                else:
                    tolerance[cycle%14] = tolerance[cycle%14] + 1
                print(tolerance)
                tolerance_changed_flag = True
                cycle += 1
    
    return tolerance_changed_list, out_name_list


def get_msa_by_regions_length_substitution(
    file_path, 
    output_dir, 
    tolerance=[0],
    chain_type="H",
    use_precomputed_msas=False,
    fold_type="openfold",
    chain_id="A",
    database="uniref90",
    minmin_seqs=1000,
    maxmin_seqs=10000,
    database_index=0,
    sub_length_max=None,
):
    """
    Finds sequences in the replacement database that meet the specified length requirements.

    Args:
        file_path: Path to the input FASTA file.
        output_dir: Path to the output directory.
        tolerance: Tolerance for the sequence lengths of various regions.
        chain_type: The type of the chain.
        use_precomputed_msas: Whether to skip sequences that have already been computed.
        fold_type: The type of fold used for docking.
        chain_id: AlphaFold's chain ID.
        database: The type of database.
        minmin_seqs: Minimum number of sequences.
        maxmin_seqs: Maximum number of sequences.
        database_index: Index of the database being used.
        sub_length_max: Maximum length in the replacement database.
    """
    assert chain_type in ["H", "L"]
    names_list, seqs_list = read_fasta_file(file_path)
    base_name = os.path.basename(file_path).split(".")[0].rstrip("_heavy").rstrip("_light")
    tolerance_changed_list = [] # Used to determine if there are differences in sequence lengths; if there are, realignment is necessary.
    out_name_list = []
    
    for index, seq in enumerate(seqs_list):
        target_names = []
        tolerance_changed_flag = True
        cycle = 0
        
        # 不同的fold_type对应不同的输出路径
        if fold_type == "openfold":
            temp_output_dir = os.path.join(output_dir, names_list[index].rstrip("_").lstrip(">"))
        elif fold_type == "alphafold":
            temp_output_dir = "{}/{}/msas/{}".format(output_dir, base_name, chain_id)
        else:
            raise ValueError("fold_type must be openfold or alphafold")

        if not os.path.exists(temp_output_dir):
            os.makedirs(temp_output_dir)

        # Path for storing the results. 'database' refers to the replacement database.
        output_path = os.path.join(temp_output_dir, "{}_hits.a3m.tmp".format(database))
            
        if use_precomputed_msas and os.path.exists(output_path.rstrip("tmp").rstrip(".")):
            tolerance_changed_list.append(False)
            out_name_list.append(output_path)
            continue
        
        regioned_seq = seq.replace("-", "")
        target_lengths = [len(region) for region in regioned_seq.split("*")]
        seq_without_region = regioned_seq.replace("*", "")
        # tolerance[5] += int(target_lengths[5] * 0.1)
        
        upper_count_list = [0 for _ in range(len(target_lengths))] # Used to store the count of regions whose lengths exceed the maximum length in the database.
        if sub_length_max is not None:
            for i in range(len(target_lengths)):
                if target_lengths[i] > sub_length_max[i]:
                    region_diff = target_lengths[i] - sub_length_max[i]
                    upper_count_list[i] = region_diff
            target_lengths = [min(target_lengths[i], sub_length_max[i]) for i in range(len(target_lengths))]
            
        while len(target_names) < minmin_seqs and cycle < 50:
            if chain_type == "H":
                target_seqs = find_sequences(
                    seqs_df=heavy_seqs_sub_list[database_index], 
                    length_df=heavy_lengths_sub_list[database_index], 
                    target_lengths=target_lengths, 
                    tolerance=tolerance)
            else:
                target_seqs = find_sequences(
                    seqs_df=light_seqs_sub_list[database_index], 
                    length_df=light_lengths_sub_list[database_index], 
                    target_lengths=target_lengths, 
                    tolerance=tolerance)
                
            target_seqs = target_seqs.to_numpy().tolist()
            if len(target_seqs) >= minmin_seqs or cycle == 49:
                if len(target_seqs) > maxmin_seqs:
                    target_seqs = [seq for seq in target_seqs if len(regioned_seq.split("*")[5]) == len(seq[5]) and hamming_distance(regioned_seq.split("*")[5], seq[5]) <= int(target_lengths[5] * 0.8 * 2)]
                target_seqs = target_seqs[:maxmin_seqs]
                target_seqs = ["".join(seq) for seq in target_seqs]
                target_seqs = [seq_without_region] + target_seqs
                target_names = [">seq_{}".format(i) for i in range(len(target_seqs))]
                
                write_fasta_file(target_names, target_seqs, output_path)
                tolerance_changed_list.append(tolerance_changed_flag)
                out_name_list.append(output_path)
                break
            else:
                if sum(upper_count_list) > 0:
                    for i, count in enumerate(upper_count_list):
                        if count > 0:
                            tolerance[i] = tolerance[i] + 1
                            upper_count_list[i] = count - 1
                            break
                else:
                    tolerance[cycle%7] = tolerance[cycle%7] + 1
                tolerance_changed_flag = True
                cycle += 1
                print(tolerance)

    return tolerance_changed_list, out_name_list


def get_msa_by_regions_length_unpaired(
    file_path, 
    output_dir, 
    database_lengths, 
    hamming_tolerance=0, 
    chain_type="H", 
    use_precomputed_msas=False, 
    database_path=None, 
    fold_type="openfold", 
    chain_id="A", 
    replace_database="single_hits",
    minmin_seqs=1000,
    maxmin_seqs=50000,
):
    """
    Finds sequences in the non-paired sequence database that meet the specified length requirements.

    Args:
        file_path: Path to the input FASTA file.
        output_dir: Path to the output directory.
        database_lengths: CDR lengths in the database, format: {'cdr3_length': ['cdr1_length', 'cdr2_length', 'path']}
        hamming_tolerance: Tolerance for Hamming distance when comparing sequences of various regions.
        chain_type: The type of the chain.
        use_precomputed_msas: Whether to skip sequences that have already been computed.
        database_path: Path to the database.
        fold_type: The type of fold used for docking.
        chain_id: AlphaFold's chain ID.
        replace_database: The replacement database.
        minmin_seqs: Minimum number of sequences.
        maxmin_seqs: Maximum number of sequences.
    """
    assert chain_type in ["H", "L"]
    
    names_list, seqs_list = read_fasta_file(file_path)
    base_name = os.path.basename(file_path).split(".")[0].rstrip("_unpaired_heavy").rstrip("_unpaired_light")
    out_name_list = []
    tolerance_changed_list = []
    
    for index, seq in enumerate(seqs_list):
        if fold_type == "openfold":
            temp_output_dir = os.path.join(output_dir, names_list[index].rstrip("_").lstrip(">"))
        elif fold_type == "alphafold":
            temp_output_dir = "{}/{}/msas/{}".format(output_dir, base_name, chain_id)
        else:
            raise ValueError("fold_type must be openfold or alphafold")

        if not os.path.exists(temp_output_dir):
            os.makedirs(temp_output_dir)

        output_path = os.path.join(temp_output_dir, "{}_hits.a3m.tmp".format(replace_database))

        if use_precomputed_msas and os.path.exists(output_path.rstrip("tmp").rstrip(".")):
            out_name_list.append(output_path)
            tolerance_changed_list.append(False)
            continue
        
        regioned_seq = seq.replace("-", "")
        seq_without_region = regioned_seq.replace("*", "")
        target_lengths = [len(region) for region in regioned_seq.split("*")]
        
        query_cdr1_length = target_lengths[1]
        query_cdr2_length = target_lengths[3]
        query_cdr3_length = target_lengths[5]
        query_cdr3 = regioned_seq.split("*")[5]
        
        # Appropriately increase the minimum number of sequences based on the CDR3 length.
        if query_cdr3_length > 10:
            minmin_seqs = minmin_seqs * query_cdr3_length // 10 * 4

        closest_lengths = [query_cdr1_length, query_cdr2_length, query_cdr3_length]
        
        closest_lengths_list = [] # Used to store the sequence length(s) closest to the query sequence CDRs.
        closest_lengths_list = cdrs_length_match(
            database_lengths=database_lengths, 
            query_length=[query_cdr1_length, query_cdr2_length, query_cdr3_length], 
            chaintype=chain_type
            )
        
        closest_idx = 0
        target_seqs_list = []
        while len(target_seqs_list) < minmin_seqs:
            if closest_idx >= len(closest_lengths_list):
                break
            
            closest_lengths = closest_lengths_list[closest_idx][0].split("_")
            target_database = os.path.join(database_path, closest_lengths_list[closest_idx][0])
            
            with open(target_database, "r") as f:
                lines_list = f.read().splitlines()
                target_names_list_temp = lines_list[0::2]
                target_seqs_list_temp = lines_list[1::2]
            
            if query_cdr3_length == int(closest_lengths[3]) and len(target_seqs_list_temp) >= minmin_seqs*10:
                hamming_distance_cutoff = int(target_lengths[5] * hamming_tolerance)
                target_seqs_list.extend([target_seqs_list_temp[index] for index, t_seq in enumerate(target_names_list_temp) if len(t_seq.lstrip(">")) == query_cdr3_length and hamming_distance(t_seq.lstrip(">"), query_cdr3) <= hamming_distance_cutoff])
            else:
                target_seqs_list.extend(target_seqs_list_temp)
                if len(target_seqs_list) > maxmin_seqs:
                    target_seqs_list = target_seqs_list[:maxmin_seqs]
                    break
            
            while len(target_seqs_list) > maxmin_seqs:
                hamming_tolerance = hamming_tolerance * 0.9
                hamming_distance_cutoff = int(target_lengths[5] * hamming_tolerance)
                target_seqs_list = [target_seqs_list_temp[index] for index, t_seq in enumerate(target_names_list_temp) if len(t_seq.lstrip(">")) == query_cdr3_length and hamming_distance(t_seq.lstrip(">"), query_cdr3) <= hamming_distance_cutoff]
                if hamming_tolerance < 0.1:
                    break

            closest_idx += 1
        
        target_seqs_list = [seq_without_region] + target_seqs_list
        names_list = [">seq_{}".format(i) for i in range(len(target_seqs_list))]
        
        write_fasta_file(names_list, target_seqs_list, output_path)
        out_name_list.append(output_path)
        tolerance_changed_list.append(True)
    
    return  tolerance_changed_list, out_name_list
    


def get_msa(args, substitute=False):
    cpus = args.cpus
    fasta_dir = args.fasta_dir
    tmp_dir = args.temp_dir    
    output_dir = args.output_dir
    use_precomputed_msas = args.use_precomputed_msas
    fold_type = args.fold_type
    fasta_names = os.listdir(fasta_dir)
    fasta_path = [os.path.join(fasta_dir, name) for name in fasta_names]
    
    if fold_type == "openfold":
        if args.use_precomputed_alignments is None:
            out_alignments_dir = os.path.join(output_dir, "alignments")
        else:
            out_alignments_dir = args.use_precomputed_alignments
    elif fold_type == "alphafold":
        out_alignments_dir = output_dir
    else:
        raise ValueError("fold_type must be openfold or alphafold")
    
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir)

    # Preload the database to avoid duplicate reads.
    global fv_seqs, fv_lengths
    fv_seqs, fv_lengths = read_data_from_pickle(fv_length_database_path["paired"])
    # Get the maximum value of each column in fv_length
    fv_lengths_max = fv_lengths.max(axis=0).tolist()
    # Find the upper x-quantile for each column of fv_length
    fv_lengths_q3 = fv_lengths.quantile(0.999).tolist()
    
    if substitute == True:
        global heavy_seqs_sub_list, heavy_lengths_sub_list, light_seqs_sub_list, light_lengths_sub_list
        for scheme, database_path in heavy_length_databases.items():
            heavy_seqs, heavy_lengths = read_data_from_pickle(database_path)
            heavy_seqs_sub_list.append(heavy_seqs)
            heavy_lengths_sub_list.append(heavy_lengths)

        for scheme, database_path in light_length_databases.items():
            light_seqs, light_lengths = read_data_from_pickle(database_path)
            light_seqs_sub_list.append(light_seqs)
            light_lengths_sub_list.append(light_lengths)
    
    unpaired_database_length_dict = {}
    current_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])
    for scheme, databse_path in unpaired_database_path.items():
        # Get the path of this file
        unpaired_database_files = os.listdir(os.path.join(current_path , databse_path))
        cdr3_length = [name.split("_")[0] + "_" + name.split("_")[3].split(".")[0] for name in unpaired_database_files]
        cdr1_2_path = [[int(name.split("_")[1]), int(name.split("_")[2]), name] for name in unpaired_database_files]
        # Use cdr3_length as the key, and cdr1_2_length as the value. If the cdr3_length is the same, append cdr1_2_length to the value.
        temp_length_dict = {}
        for cdr3, cdr1_2 in zip(cdr3_length, cdr1_2_path):
            file_info = UnpairedFasta(cdr1_2[0], cdr1_2[1], cdr1_2[2])
            if cdr3 not in temp_length_dict.keys():
                temp_length_dict[cdr3] = [file_info]
            else:
                temp_length_dict[cdr3].append(file_info)
        unpaired_database_length_dict[scheme] = temp_length_dict

    for file_path in fasta_path:
        original_seqs_name, original_seqs = read_fasta_file(file_path)
        seq_to_dict = {} # Used to specify the output folder name for AlphaFold multiple sequence alignment.
        for chain_id, seq_name in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", original_seqs_name):
            seq_to_dict[seq_name.strip().strip("_").rstrip(">")] = chain_id
        
        print("\nbuild alignments for single ...")
        # Generate MSA for the single-chain database
        for index, (scheme, database_path) in enumerate(unpaired_database_path.items()):
            time_start = time()
            # Generate heavy chain MSA
            temp_out_path = os.path.join(tmp_dir, "{}_unpaired_heavy.fasta".format(os.path.basename(file_path).split(".")[0]))
            regioned_file_path = temp_out_path + ".temp.txt"
            run_alignment(fas_file=file_path, 
                        out_path=temp_out_path, 
                        scheme=scheme, 
                        chain="H", 
                        get_regioned_file=True,
                        cpus=cpus,
                        cutoff=60,
            )
            if os.path.exists(regioned_file_path):
                heavy_names, _ = read_fasta_file(temp_out_path)
                base_name = os.path.basename(regioned_file_path).split(".")[0]
                # Get sequences related to the query sequence and write them to a file.
                tolerance_changed_list, out_temp_name_list = get_msa_by_regions_length_unpaired(
                    file_path=regioned_file_path,
                    output_dir=out_alignments_dir,
                    hamming_tolerance=0.5*2,
                    fold_type=fold_type,
                    chain_type="H",
                    use_precomputed_msas=use_precomputed_msas,
                    database_path=os.path.join(current_path, database_path),
                    database_lengths=unpaired_database_length_dict[scheme],
                    chain_id=seq_to_dict[heavy_names[0].strip("_").rstrip(">")])
                
                out_name_list = []
                for path in out_temp_name_list:
                    dir_path = os.path.dirname(path)
                    file_name = os.path.basename(path).rstrip("tmp").rstrip(".") # Directly stripping ".tmp" can lead to an incorrect filename, possibly a Python bug.
                    out_path = os.path.join(dir_path, file_name)
                    out_name_list.append(out_path)
                
                # Align the retrieved sequences
                for index, flag in enumerate(tolerance_changed_list):
                    if flag:
                        run_alignment(fas_file=out_temp_name_list[index], out_path=out_name_list[index], scheme=scheme, chain="H", cpus=cpus)
                        msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[index])
                        write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[index])
                        
                    if os.path.exists(out_temp_name_list[index]):
                        os.remove(out_temp_name_list[index])
                        
                # Delete temporary file
                if os.path.exists(regioned_file_path):
                    os.remove(regioned_file_path)
                if os.path.exists(temp_out_path):
                    os.remove(temp_out_path)

            temp_out_path = os.path.join(tmp_dir, "{}_unpaired_light.fasta".format(os.path.basename(file_path).split(".")[0]))
            regioned_file_path = temp_out_path + ".temp.txt"
            run_alignment(fas_file=file_path,
                        out_path=temp_out_path,
                        scheme=scheme,
                        chain="L",
                        get_regioned_file=True,
                        cpus=cpus
            )
            if os.path.exists(regioned_file_path):
                light_names, _ = read_fasta_file(temp_out_path)
                base_name = os.path.basename(regioned_file_path).split(".")[0]
                tolerance_changed_list, out_temp_name_list = get_msa_by_regions_length_unpaired(
                    file_path=regioned_file_path,
                    output_dir=out_alignments_dir,
                    hamming_tolerance=0.4*2,
                    fold_type=fold_type,
                    chain_type="L",
                    use_precomputed_msas=use_precomputed_msas,
                    database_path=os.path.join(current_path, database_path),
                    database_lengths=unpaired_database_length_dict[scheme],
                    chain_id=seq_to_dict[light_names[0].strip("_").rstrip(">")])
                
                out_name_list = []
                for path in out_temp_name_list:
                    dir_path = os.path.dirname(path)
                    file_name = os.path.basename(path).rstrip("tmp").rstrip(".")
                    out_path = os.path.join(dir_path, file_name)
                    out_name_list.append(out_path)
                    
                for index, flag in enumerate(tolerance_changed_list):
                    if flag:
                        run_alignment(fas_file=out_temp_name_list[index], out_path=out_name_list[index], scheme=scheme, chain="L", cpus=cpus)
                        msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[index])
                        write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[index])
                    if os.path.exists(out_temp_name_list[index]):
                        os.remove(out_temp_name_list[index])
                
                if os.path.exists(regioned_file_path):
                    os.remove(regioned_file_path)
                if os.path.exists(temp_out_path):
                    os.remove(temp_out_path)
            time_end = time()
            print("# Single finished, time cost: {:.2f}s, {}".format(time_end - time_start, file_path))
        
        if len(original_seqs) == 2:
            print("\nbuild alignments for pair ...")
            for scheme, database_path in fv_length_database_path.items():
                time_start = time()
                out_heavy_path = os.path.join(tmp_dir, "{}_{}_heavy.fasta".format(os.path.basename(file_path).split(".")[0], scheme))
                regioned_heavy_file_path = out_heavy_path + ".temp.txt"
                run_alignment(fas_file=file_path, 
                            out_path=out_heavy_path, 
                            scheme="chothia", 
                            chain="H", 
                            get_regioned_file=True,
                            cpus=cpus
                )
                
                out_light_path = os.path.join(tmp_dir, "{}_{}_light.fasta".format(os.path.basename(file_path).split(".")[0], scheme))
                regioned_light_file_path = out_light_path + ".temp.txt"
                run_alignment(fas_file=file_path,
                            out_path=out_light_path,
                            scheme="chothia",
                            chain="L",
                            get_regioned_file=True,
                            cpus=cpus
                )
                
                if os.path.exists(regioned_heavy_file_path) and os.path.exists(regioned_light_file_path):
                    merged_fasta = os.path.join(tmp_dir, "{}_merged.fasta".format(os.path.basename(file_path).split(".")[0], scheme))
                    merge_fasta_file(regioned_heavy_file_path, regioned_light_file_path, merged_fasta)
                    heavy_names, _ = read_fasta_file(out_heavy_path)
                    light_names, _ = read_fasta_file(out_light_path)
                    chain_id = []
                    for h_name, l_name in zip(heavy_names, light_names):
                        chain_id.append([seq_to_dict[h_name.strip("_").rstrip(">")], 
                                        seq_to_dict[l_name.strip("_").rstrip(">")]])
                    
                    tolerance_changed_list, out_temp_name_list = get_msa_by_regions_length_paired(file_path=merged_fasta, 
                                                output_dir=out_alignments_dir, tolerance=[16, 0, 2, 0, 5, 0, 10, 16, 0, 2, 0, 2, 0, 10], 
                                                fold_type=fold_type,
                                                use_precomputed_msas=use_precomputed_msas,
                                                database_path=database_path,
                                                chain_id=chain_id,
                                                fv_lengths_max=fv_lengths_max,
                                                fv_lengths_qx=fv_lengths_q3,)
                    
                    out_name_list = []
                    for pair_list in out_temp_name_list:
                        temp_list = []
                        for pair in pair_list:
                            dir_path = os.path.dirname(pair)
                            file_name = os.path.basename(pair)
                            file_name = file_name.rstrip("tmp").rstrip(".")
                            out_path = os.path.join(dir_path, file_name)
                            temp_list.append(out_path)
                        out_name_list.append(temp_list)
                        
                    for index, flag in enumerate(tolerance_changed_list):
                        if flag:
                            run_alignment(fas_file=out_temp_name_list[index][0], out_path=out_name_list[index][0], scheme="chothia", chain="H", cpus=cpus)
                            msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[index][0])
                            msa_names_list = [">tr|A0A8T2NDK6|A0A8T2NDK6_{}/H{}-L{}".format(index, index, index) for index, name in enumerate(msa_names_list, 1)]
                            msa_names_list[0] = ">Heavy_chain"
                            write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[index][0])
                            
                            run_alignment(fas_file=out_temp_name_list[index][1], out_path=out_name_list[index][1], scheme="chothia", chain="L", cpus=cpus)
                            msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[index][1])
                            msa_names_list = [">tr|A0A8T2NDK6|A0A8T2NDK6_{}/L{}-H{}".format(index, index, index) for index, name in enumerate(msa_names_list, 1)]
                            msa_names_list[0] = ">Light_chain"
                            write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[index][1])
                        if os.path.exists(out_temp_name_list[index][0]):
                            os.remove(out_temp_name_list[index][0])
                        if os.path.exists(out_temp_name_list[index][1]):
                            os.remove(out_temp_name_list[index][1])
                            
                    if os.path.exists(merged_fasta):
                        os.remove(merged_fasta)
                    
                elif os.path.exists(regioned_heavy_file_path) and not os.path.exists(regioned_light_file_path):
                    print("# ERROR: {}: Not exists light chain".format(file_path))
                elif not os.path.exists(regioned_heavy_file_path) and os.path.exists(regioned_light_file_path):
                    print("# ERROR: {}: Not exists heavy chain".format(file_path))
                else:
                    print("# ERROR: {}: Not exists heavy chain and light chain".format(file_path))
                if os.path.exists(regioned_heavy_file_path):
                    os.remove(regioned_heavy_file_path)
                if os.path.exists(regioned_light_file_path):
                    os.remove(regioned_light_file_path)
                if os.path.exists(out_heavy_path):
                    os.remove(out_heavy_path)
                if os.path.exists(out_light_path):
                    os.remove(out_light_path)
                time_end = time()
                print("# Pair finished, time cost: {:.2f}s, {}".format(time_end - time_start, file_path))
        
        if substitute == True:
            print("\nbuild alignments for uniref90 ...")
            time_start = time()
            sub_length_max = heavy_lengths_sub_list[0].max(axis=0).tolist()
            for index, (scheme, database_path) in enumerate(heavy_ab_database_path.items()):
                # database_name = os.path.basename(database_path).split("_")[1] # 默认数据库的名字为"CH_uniref90_hits.pkl"因此这样处理
                database_name = "uniref90"
                scheme = "chothia"
                temp_out_path = os.path.join(tmp_dir, "{}_heavy.fasta".format(os.path.basename(file_path).split(".")[0]))
                regioned_file_path = temp_out_path + ".temp.txt"
                run_alignment(fas_file=file_path, 
                            out_path=temp_out_path, 
                            scheme=scheme, 
                            chain="H", 
                            get_regioned_file=True,
                            cpus=cpus
                )
                if os.path.exists(regioned_file_path):
                    heavy_names, _ = read_fasta_file(temp_out_path)
                    base_name = os.path.basename(regioned_file_path).split(".")[0]
                    tolerance_changed_list, out_temp_name_list = get_msa_by_regions_length_substitution(file_path=regioned_file_path, 
                                            output_dir=out_alignments_dir, tolerance=[10, 0, 0, 0, 0, 0, 5], 
                                            chain_type="H", 
                                            use_precomputed_msas=use_precomputed_msas,
                                            fold_type=fold_type,
                                            chain_id=seq_to_dict[heavy_names[0].strip("_").rstrip(">")],
                                            database=database_name,
                                            database_index=index,
                                            sub_length_max=sub_length_max,
                    )
                    
                    out_name_list = []
                    for path in out_temp_name_list:
                        dir_path = os.path.dirname(path)
                        file_name = os.path.basename(path).rstrip("tmp").rstrip(".")
                        out_path = os.path.join(dir_path, file_name)
                        out_name_list.append(out_path)
                    
                    for index, flag in enumerate(tolerance_changed_list):
                        if flag:
                            run_alignment(fas_file=out_temp_name_list[index], out_path=out_name_list[index], scheme=scheme, chain="H", cpus=cpus)
                            msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[index])
                            write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[index])
                        if os.path.exists(out_temp_name_list[index]):
                            os.remove(out_temp_name_list[index])
                            
                    if os.path.exists(regioned_file_path): 
                        os.remove(regioned_file_path)
                    if os.path.exists(temp_out_path):
                        os.remove(temp_out_path)

            for index, (scheme, database_path) in enumerate(light_ab_database_path.items()):
                # database_name = os.path.basename(database_path).split("_")[1] # 默认数据库的名字为"CL_uniref90_hits.pkl"因此这样处理
                database_name = "uniref90"
                scheme = "chothia"
                temp_out_path = os.path.join(tmp_dir, "{}_light.fasta".format(os.path.basename(file_path).split(".")[0]))
                regioned_file_path = temp_out_path + ".temp.txt"
                run_alignment(fas_file=file_path,
                                out_path=temp_out_path,
                                scheme=scheme,
                                chain="L",
                                get_regioned_file=True,
                                cpus=cpus
                )
                
                if os.path.exists(regioned_file_path):
                    light_names, _ = read_fasta_file(temp_out_path)
                    tolerance_changed_list, out_temp_name_list = get_msa_by_regions_length_substitution(file_path=regioned_file_path,
                                            output_dir=out_alignments_dir, tolerance=[10, 0, 10, 0, 10, 0, 10],
                                            chain_type="L",
                                            use_precomputed_msas=use_precomputed_msas,
                                            fold_type=fold_type,
                                            chain_id=seq_to_dict[light_names[0].strip("_").rstrip(">")],
                                            database=database_name,
                                            database_index=index,
                                            sub_length_max=sub_length_max,
                    )
                    out_name_list = []
                    for path in out_temp_name_list:
                        dir_path = os.path.dirname(path)
                        file_name = os.path.basename(path).rstrip("tmp").rstrip(".")
                        out_path = os.path.join(dir_path, file_name)
                        out_name_list.append(out_path)
                    
                    for index, flag in enumerate(tolerance_changed_list):
                        if flag:
                            run_alignment(fas_file=out_temp_name_list[index], out_path=out_name_list[index], scheme=scheme, chain="L", cpus=cpus)
                            msa_names_list, msa_seqs_list = delete_msa_by_first_seq(out_name_list[index])
                            write_fasta_file(msa_names_list, msa_seqs_list, out_name_list[index])
                        if os.path.exists(out_temp_name_list[index]):
                            os.remove(out_temp_name_list[index])
                        
                    if os.path.exists(regioned_file_path):
                        os.remove(regioned_file_path)
                    if os.path.exists(temp_out_path):
                        os.remove(temp_out_path)
            time_end = time()
            print("# Sub finished, time cost: {:.2f}s, {}".format(time_end - time_start, file_path))