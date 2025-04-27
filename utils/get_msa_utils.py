import pandas as pd
from utils.database import regions, regions_fv
from utils.fasta import read_fasta_file,  save_data_to_pickle


def get_seqs_length(file_path):
    names_list, seqs_list = read_fasta_file(file_path)
    seqs_list = [seq.replace("-", "").split("*") for seq in seqs_list]
    length_list = [list(map(len, seq)) for seq in seqs_list]
    return length_list, seqs_list


def save_seq_length_to_pickle(seqs_path, output_path):
    length_list, seqs_list = get_seqs_length(seqs_path)
    seqs_df = pd.DataFrame(seqs_list, columns=regions)
    length_df = pd.DataFrame(length_list, columns=regions)
    save_data_to_pickle((seqs_df, length_df), output_path)
    

def find_sequences(
    seqs_df, 
    length_df, 
    target_df, 
    tolerance=0, 
    ):
    
    # Calculate the absolute difference between the target length and each row of the sequence data.
    diff = (length_df - target_df).abs()

    # Filter out rows where the difference is less than or equal to the tolerance.
    matched_sequences = seqs_df[(diff <= tolerance).all(axis=1)]
    
    return matched_sequences


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings"""
    if len(s1) != len(s2):
        raise ValueError("Two strings have different lengths, cannot calculate Hamming distance")

    distance = 0
    for i in range(len(s1)):
        distance += bin(ord(s1[i]) ^ ord(s2[i])).count('1')
    return distance


def split_list(list_, n):
    """将列表分割为n个子列表"""    
    n = min(n, len(list_))

    quotient = len(list_) // n
    remainder = len(list_) % n
    result = []
    for i in range(n):
        if i < remainder:
            result.append(list_[i * (quotient + 1):(i + 1) * (quotient + 1)])
        else:
            result.append(list_[i * quotient + remainder:(i + 1) * quotient + remainder])
    return result


