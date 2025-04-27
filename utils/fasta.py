import itertools
import pickle


def save_data_to_pickle(data, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)


def read_data_from_pickle(infile):
    with open(infile, 'rb') as f:
        data = pickle.load(f)
    return data


def read_fasta_file(file_path):
    with open(file_path, "r") as f:
        dat = ""
        seqs_list = []
        names_list = []
        start = False
        for line in f.readlines():
            if line[0] == ">":
                start=True
                names_list.append(line.rstrip("\n").strip(" "))
                if dat != "":
                    seqs_list.append(dat)
                    dat = ""
            else:
                if start:
                    dat += line.rstrip("\n")
        if dat != "":
            seqs_list.append(dat.strip("\n"))
        
        if len(seqs_list) != len(names_list):
            raise IndexError("name count not equal seq count!")
        
        return names_list, seqs_list


def write_fasta_file( names_list, seqs_list, out_path):
    with open(out_path, "w") as f:
        f.write("\n".join(list(itertools.chain(*zip(names_list, seqs_list)))))
        

def merge_fasta_file(left_path, right_path, output_path):
    """Merge two fasta files into one file. 
    And the sequence in the left file will be added 
    to the right sequence with a "*" between them."""
    left_name_list, left_seq_list = read_fasta_file(left_path)
    right_name_list, right_seq_list = read_fasta_file(right_path)
    
    if len(left_seq_list) != len(right_seq_list):
        raise Exception("The length of two fasta files are not equal.")
    result_seq_list = [left + "*" + right for right, left in zip(right_seq_list, left_seq_list)]
    if len(result_seq_list) != len(right_name_list):
        raise Exception("The length of result fasta file is not equal to the length of name list.")
    
    out_name_list = [left_name_list[idx] + "|" + name.lstrip(">") for idx, name in enumerate(right_name_list)]
    write_fasta_file(out_name_list, result_seq_list, output_path)