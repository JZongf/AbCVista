import os
import json
import shutil
import numpy as np
from utils.fasta import read_fasta_file, write_fasta_file
from utils.get_antibody_region import REGION_NAME


def mask_msa(
    msa: list,
    msak_start: int,
    mask_cutlen: int,
    mask_width: int,
    mask_step: int,
    mask_value: str,
):
    n_seqs = len(msa)
    n_length = len(msa[0])

    for i in range(n_seqs):
        mask_end = msak_start + mask_width
        back_width = 0
        if mask_end > n_length - 1:
            back_width = mask_end - n_length
            mask_end = n_length - 1

        if back_width > 0:
            msa[i][mask_cutlen : back_width + mask_cutlen] = mask_value * back_width

        msa[i][msak_start:mask_end] = mask_value * (mask_end - msak_start)
        if (i + 1) % 128 == 0:
            msak_start += mask_step

        if msak_start > n_length:
            msak_start = mask_cutlen

    return msa


def random_mask_msa(
    msa: np.ndarray,
    mask_count: int,
    mask_value: str,
    min_mask_pos: int,
):
    # Get the shape of the MSA
    num_sequences, sequence_length = msa.shape

    # Iterate through each sequence
    for i in range(num_sequences):
        # Generate mask_count random positions greater than or equal to min_mask_pos.
        mask_positions = np.random.choice(
            range(min_mask_pos, sequence_length), size=mask_count, replace=False
        )
        # Apply masking at these positions
        msa[i, mask_positions] = mask_value

    return msa


def get_seq_part(
    ori_seq,
    input_file,
    output_file,
    full_seq,
):
    """
    Reconstruct the MSA file by removing the antibody part.
    args:
    ori_seq: str, the original sequence.
    input_file: str, the input file path.
    output_file: str, the output file path.
    full_seq: str, the full sequence of query.
    """

    target_start = full_seq.find(ori_seq)
    target_end = target_start + len(ori_seq)
    names, seqs = read_fasta_file(input_file)
    seqs = [seq[target_start:target_end] for seq in seqs]
    seqs[0] = ori_seq
    write_fasta_file(names, seqs, output_file)


def rebuild_msa(
    input_file,
    output_file,
    script_path=None,
    python_path=None,
):
    """
    Reconstruct the MSA file by protein language model.
    args:
    input_file: str, the input file path.
    output_file: str, the output file path.
    model_path: str, the model path.
    python_path: str, the environment path of python.
    """
    
    cmd = f"{python_path} {script_path} --input_msa {input_file} --output_msa {output_file}"
    os.system(cmd)


def get_msa(args):
    """
    Get the MSA for the antigen. 
    The MSA includes the uniref90_hits.a3m and paired_hits.a3m files.
    """
    fasta_dir = args.fasta_dir
    output_dir = os.path.join(args.output_dir, "alignments")

    exist_chains = set(
        [
            chain
            for chain in os.listdir(output_dir)
            if os.path.exists(os.path.join(output_dir, chain, "region_index.json"))
        ]
    )

    for fasta in os.listdir(fasta_dir):
        names, seqs = read_fasta_file(os.path.join(fasta_dir, fasta))
        names = [name.lstrip(">") for name in names]

        if len(names) > 1:
            paired_names = []
            paired_seqs = []
            for idx, name in enumerate(names):
                if name in exist_chains:
                    paired_names, paired_seqs = read_fasta_file(
                        os.path.join(output_dir, name, "paired_hits.a3m")
                    )
                    break

        antigen_idx = 0
        for idx, name in enumerate(names):
            if name not in exist_chains:
                if not os.path.exists(os.path.join(output_dir, name)):
                    os.mkdir(os.path.join(output_dir, name))
                write_fasta_file(
                    [f">Antigen_{antigen_idx}"],
                    [seqs[idx]],
                    os.path.join(output_dir, name, "uniref90_hits.a3m"),
                )

                if len(names) > 1:
                    out_names = [">Antigen_{}".format(antigen_idx)]
                    out_names += [
                        f">tr|A0A8T2NDK6|A0A8T2NDK6_{i+1}/O{i+1}-H{i+1}"
                        for i in range(1, len(paired_names))
                    ]
                    out_seqs = [seqs[idx] for _ in range(1, len(paired_names))]

                    write_fasta_file(
                        out_names,
                        out_seqs,
                        os.path.join(output_dir, name, "paired_hits.a3m"),
                    )
                    antigen_idx += 1


def region_padding(
    region_file,
    front_len,
    back_len,
):
    """
    compute the padding length for the region file.
    """
    with open(region_file, "r") as f:
        region_dict = json.load(f)
    
    for region in REGION_NAME:
        region_dict[region][0] += front_len
        region_dict[region][1] += front_len
    
    region_dict["FRONT"] = [0, front_len]
    region_dict["BACK"] = [region_dict["FR4"][1], back_len]
    region_dict["length"] = region_dict["BACK"][1]
    
    with open(region_file, "w") as f:
        json.dump(region_dict, f, indent=4)


def msa_padding(
    msa_file,
    padding_seq,
    region_file=None,
):
    """
    Padding the MSA file with the padding sequence.
    args:
    msa_file: str, the input MSA file path.
    padding_seq: str, the padding sequence.
    region_file: str, the input region file path.
    """
    padding_seq = padding_seq.replace("-", "")
    with open(msa_file, "r") as f:
        lines = f.read().splitlines()
        names = lines[::2]
        seqs = lines[1::2]

    loc_start = padding_seq.find(seqs[0].replace("-", ""))
    if len(padding_seq) > len(seqs[0]):
        loc_end = loc_start + len(seqs[0])
        # front_padding = padding_seq[:loc_start]
        # back_padding = padding_seq[loc_end:]
        
        front_padding = "-" * loc_start
        back_padding = "-" * (len(padding_seq) - loc_end)
        
        padded_seqs = [front_padding + seq + back_padding for seq in seqs]
        padded_seqs[0] = padding_seq
        
        with open(msa_file, "w") as f:
            for name, seq in zip(names, padded_seqs):
                f.write(name + "\n" + seq + "\n")
        
    if os.path.exists(region_file):
        region_padding(region_file, loc_start, len(padding_seq))


def msa_gap_padding(
    msa_file,
    region_file,
):
    """
    Replace the padding sequence with the gap sequence.
    args:
    msa_file: str, the input MSA file path.
    padding_seq: str, the padding sequence.
    region_file: str, the input region file path.
    """
    with open(msa_file, "r") as f:
        lines = f.read().splitlines()
        names = lines[::2]
        seqs = lines[1::2]

    seqs = np.array([list(seq) for seq in seqs], dtype=object)
    
    with open(region_file, "r") as f:
        region_dict = json.load(f)
    
    tg_seq = seqs[0].copy()
    seqs[:, region_dict["FRONT"][0]:region_dict["FRONT"][1]] = "-"
    seqs[:, region_dict["BACK"][0]:region_dict["BACK"][1]] = "-"
    
    seqs[0] = tg_seq
    with open(msa_file, "w") as f:
        for name, seq in zip(names, seqs):
            f.write(name + "\n" + "".join(seq) + "\n")


def padding_msas(
    args,
    gap_padding=False
):
    """
    padding the antibody MSA files.
    """
    fasta_dir = args.fasta_dir
    output_dir = os.path.join(args.output_dir, "alignments")
    for fasta in os.listdir(fasta_dir):
        names, seqs = read_fasta_file(os.path.join(fasta_dir, fasta))
        names = [name.lstrip(">") for name in names]
        
        for i, name in enumerate(names):
            alignment_dir = os.path.join(output_dir, name)
            if not os.path.exists(os.path.join(alignment_dir, "region_index.json")):
                continue
            
            if os.path.exists(alignment_dir):
                for file in os.listdir(alignment_dir):
                    if file.endswith(".a3m"):
                        msa_file = os.path.join(alignment_dir, file)
                        if gap_padding:
                            msa_gap_padding(msa_file, os.path.join(alignment_dir, "region_index.json"))
                        else:
                            msa_padding(msa_file, seqs[i], os.path.join(alignment_dir, "region_index.json"))


def filter_array(arr, cutoff=3):
    result = []
    start = 0
    end = 0

    while end < len(arr):
        # Find the starting position of the next continuous region
        while end < len(arr) and (end == start or arr[end] != arr[start] + end - start):
            start = end
            end += 1

        # Record the ending position of this continuous region
        while end < len(arr) and arr[end] == arr[start] + end - start:
            end += 1

        # If the length of this continuous region is greater than or equal to 3, add it to the result array.
        if end - start > 3:
            result.extend(arr[start:end])

    return np.array(result).astype(int)


def merge_msas(args, antibody_list, alignment_dir):

    # merge antibody
    for ab in antibody_list:
        inner_paired_abs = ab.get_inchain_antibodies()
        for ori_name, inner_abs in inner_paired_abs.items():
            full_seq = inner_abs[0].full_seq
            if len(inner_abs) == 1:
                continue
            
            tmp_align_dir = os.path.join(alignment_dir, ori_name)
            if not os.path.exists(tmp_align_dir):
                os.mkdir(tmp_align_dir)
            
            inner_paired_dict = {}
            outter_paired_list = []
            single_files_dict = {}
            for inner_ab in inner_abs:
                for file in os.listdir(os.path.join(alignment_dir, inner_ab.name)):
                    if file.endswith(".a3m"):
                        if file.startswith("inner"):
                            if file not in inner_paired_dict:
                                inner_paired_dict[file] = []
                            inner_paired_dict[file].append(os.path.join(alignment_dir, inner_ab.name, file))
                        elif file.startswith("paired"):
                            outter_paired_list.append(os.path.join(alignment_dir, inner_ab.name, file))
                        else:
                            if file not in single_files_dict:
                                single_files_dict[file] = []
                            single_files_dict[file].append(os.path.join(alignment_dir, inner_ab.name, file))
            
            # merge inner paired
            for file, file_list in inner_paired_dict.items():
                if len(file_list) == 2:
                    out_file = os.path.join(tmp_align_dir, file)
                    seqs_list = []
                    names_list = []
                    for file in file_list:
                        with open(file, "r") as f:
                            lines = f.read().splitlines()
                            names = lines[::2]
                            seqs = lines[1::2]
                            seqs_list.append(seqs)
                            names_list.append(names)
                    
                    # get min count of seqs_list
                    min_count = min([len(seqs) for seqs in seqs_list])
                    seqs_list = [seqs[:min_count] for seqs in seqs_list]
                    names_list = [names[:min_count] for names in names_list]
                    
                    seq1_start = full_seq.find(seqs_list[0][0].replace("-", ""))
                    seq1_end = len(seqs_list[0][0].replace("-", "")) + seq1_start
                    seq2_start = full_seq.find(seqs_list[1][0].replace("-", ""))
                    linker = '-' * (seq2_start - seq1_end)
                    seq2_back = len(full_seq) - seq2_start - len(seqs_list[1][0].replace("-", ""))
                    
                    front_linker = '-' * seq1_start
                    back_linker = '-' * seq2_back
                    
                    connect_seq = [front_linker + seqs_list[0][i] + linker + seqs_list[1][i] + back_linker for i in range(min_count)]
                    connect_seq[0] = full_seq
                    
                    with open(out_file, "w") as f:
                        for i, seq in enumerate(connect_seq):       
                            f.write(f">seq{i}\n{seq}\n")

            # merge outter paired
            if os.path.exists(os.path.join(tmp_align_dir, "paired_hits.a3m")):
                os.remove(os.path.join(tmp_align_dir, "paired_hits.a3m"))
            
            count = 0
            for idx, file in enumerate(sorted(outter_paired_list, reverse=True)):
                with open(os.path.join(tmp_align_dir, "paired_hits.a3m"), "a") as paired_f:
                    if idx == 0:
                        paired_f.write(f">Target\n{full_seq}\n")
                    with open(file, "r") as f:
                        lines = f.read().splitlines()
                        names = lines[::2]
                        seqs = lines[1::2]
                        
                        front_gap = full_seq.find(seqs[0].replace("-", ""))
                        back_gap = len(full_seq) - front_gap - len(seqs[0].replace("-", ""))
                        
                        front_linker = '-' * front_gap
                        back_linker = '-' * back_gap
                        
                        for i, seq in enumerate(seqs[1:], start=1):
                            paired_f.write(f"{names[i]}\n{ front_linker + seq + back_linker}\n")
                            
            # merge single files
            for file, file_list in single_files_dict.items():
                with open(os.path.join(tmp_align_dir, file), "w") as f:
                    f.write(f">Target\n{full_seq}\n")
                    for file in file_list:
                        with open(file, "r") as f2:
                            lines = f2.read().splitlines()
                            seqs = lines[1::2]
                            names = lines[::2]
                            
                            front_gap = full_seq.find(seqs[0].replace("-", ""))
                            back_gap = len(full_seq) - front_gap - len(seqs[0].replace("-", ""))
                            
                            for i, seq in enumerate(seqs[1:], start=1):
                                f.write(f">seq{i}\n{ '-' * front_gap + seq + '-' * back_gap}\n")
                                
    # rebuild linker
    for ab in antibody_list:
        inner_paired_abs = ab.get_inchain_antibodies()
        for ori_name, inner_abs in inner_paired_abs.items():
            if len(inner_abs) == 1:
                continue
            
            tmp_align_dir = os.path.join(alignment_dir, ori_name)
            
            for file in os.listdir(tmp_align_dir):
                if file.endswith(".a3m"):
                    with open(os.path.join(tmp_align_dir, file), "r") as f:
                        lines = f.read().splitlines()
                        names = lines[::2]
                        seqs = lines[1::2]

                    # to array
                    seqs = np.array([list(seq) for seq in seqs], dtype=object)
                    
                    # Identify columns (excluding the first row) that consist entirely of gaps.
                    gap_cols = np.all(seqs[1:] == '-', axis=0)
                    gap_cols = np.where(gap_cols)[0]
                    if gap_cols.size == 0:
                        continue
                    
                    # Index of the last column among all non-gap columns.
                    try:
                        last_col_idx = np.where(np.any(seqs[1:] != '-', axis=0))[0][-1]
                    except IndexError:
                        print(f"INFO: {file} has no non-gap column.")
                    gap_cols = gap_cols[gap_cols < last_col_idx]
                    
                    # Exclude columns from gap_cols with a continuous length less than 3.
                    gap_cols = filter_array(gap_cols, cutoff=3)
                    
                    linker = seqs[0][gap_cols]
                    for i, seq in enumerate(seqs[1:], start=1):
                        shuffled_linker = np.random.permutation(linker)
                        seqs[i][gap_cols] = shuffled_linker
                    
                    with open(os.path.join(tmp_align_dir, file), "w") as f:
                        for name, seq in zip(names, seqs):
                            f.write(name + "\n" + "".join(seq) + "\n")