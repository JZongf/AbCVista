import os
import sys
import random
from concurrent.futures import ProcessPoolExecutor
from utils.align import run_alignment, get_clonotype
from utils.fasta import read_fasta_file
import re
import pandas as pd
from itertools import chain
from itertools import combinations, zip_longest


class AntiBodySingle:
    def __init__(self, name, seq, chain_type, clonotype=None, full_seq=None, ori_name=None):
        self.name = name
        self.seq = seq
        self.full_seq = full_seq
        self.ori_name = ori_name
        self.chain_type = chain_type
        self.clonotype = clonotype

    def set_clonotype(self, clonotype):
        self.clonotype = clonotype

    def __eq__(self, value: object) -> bool:
        if isinstance(value, str):
            return self.name == value
        elif isinstance(value, AntiBodySingle):
            return self.name == value.name
        else:
            return False


class PairAntiBody:
    def __init__(
        self,
        heavy_antibody: AntiBodySingle = None,
        light_antibody: AntiBodySingle = None,
        is_bsab=False,
    ):
        self.heavy_antibody = heavy_antibody
        self.light_antibody = light_antibody
        self.is_bsab = is_bsab

    def set_heavy_antibody(self, heavy_antibody: AntiBodySingle):
        self.heavy_antibody = heavy_antibody

    def set_light_antibody(self, light_antibody: AntiBodySingle):
        self.light_antibody = light_antibody

    def get_heavy_antibody(self):
        return self.heavy_antibody

    def get_light_antibody(self):
        return self.light_antibody

    def is_paired(self):
        return self.heavy_antibody is not None and self.light_antibody is not None

    def get_all_antibodies(self):
        if self.light_antibody is None and self.heavy_antibody is not None:
            return [self.heavy_antibody]
        elif self.heavy_antibody is None and self.light_antibody is not None:
            return [self.light_antibody]
        else:
            return [self.heavy_antibody, self.light_antibody]


class AntiBody:
    def __init__(
        self,
    ):
        self.heavy_antibody = []
        self.light_antibody = []
        self.names_to_seqs = {}
        self.inner_pair_antibody = []

    def add_heavy_antibody(self, heavy_antibody: AntiBodySingle):
        self.heavy_antibody.append(heavy_antibody)

    def add_light_antibody(self, light_antibody: AntiBodySingle):
        self.light_antibody.append(light_antibody)

    def add_inner_pair_antibody(self, inner_pair_antibody: PairAntiBody):
        self.inner_pair_antibody.append(inner_pair_antibody)
    
    def add_name(self, name, seq):
        self.names_to_seqs[name] = seq
        
    def get_name(self):
        return self.names_to_seqs

    def get_heavy_antibody(self):
        return self.heavy_antibody

    def get_light_antibody(self):
        return self.light_antibody

    def is_paired(self):
        return self.heavy_antibody != [] and self.light_antibody != []

    def get_inchain_antibodies(self):
        # 获取在同一条链上的所有抗体
        id_to_antibody = {}
        for h_chain in self.heavy_antibody:
            if id_to_antibody.get(h_chain.ori_name) is None:
                id_to_antibody[h_chain.ori_name] = [h_chain]
            else:
                id_to_antibody[h_chain.ori_name].append(h_chain)

        for l_chain in self.light_antibody:
            if id_to_antibody.get(l_chain.ori_name) is None:
                id_to_antibody[l_chain.ori_name] = [l_chain]
            else:
                id_to_antibody[l_chain.ori_name].append(l_chain)

        # Sort antibodies according to their sequence order on the chain
        for antibody_list in id_to_antibody.values():
            antibody_list.sort(key=lambda x: x.full_seq.find(x.seq.replace("*", "")))
        
        return id_to_antibody

    def get_all_antibodies(self):
        if len(self.heavy_antibody) == 0 and len(self.light_antibody) == 0:
            return None
        elif len(self.heavy_antibody) == 1 and len(self.light_antibody) == 1:
            return [PairAntiBody(self.heavy_antibody[0], self.light_antibody[0])]
        else:
            # id_to_antibody = self.get_inchain_antibodies()
            # antibody_list = [val for val in id_to_antibody.values()]
            # min_col = min(len(antibody_list[i]) for i in range(len(antibody_list)))
            
            # for i in range(min_col):
            #     for j in range(len(antibody_list)-1):
            #         if antibody_list[j][i].chain_type != antibody_list[j+1][i].chain_type:
            #             antibody_heavy = antibody_list[j][i] if antibody_list[j][i].chain_type == "H" else antibody_list[j+1][i]
            #             antibody_light = antibody_list[j+1][i] if antibody_list[j][i].chain_type == "H" else antibody_list[j][i]
                        
            #             paired_list.append(PairAntiBody(antibody_heavy, antibody_light))
            
            max_len = max(len(self.heavy_antibody), len(self.light_antibody))
            heavy_antibody_list = [None] * (max_len - len(self.heavy_antibody)) + self.heavy_antibody
            light_antibody_list = [None] * (max_len - len(self.light_antibody)) + self.light_antibody
            
            result_list = []
            paired_list = []
            for hchain, lchain in zip(heavy_antibody_list, light_antibody_list):
                if hchain is not None and lchain is not None:
                    if hchain.ori_name != lchain.ori_name:
                        paired_list.append(PairAntiBody(hchain, lchain, is_bsab=False))

            if paired_list == []:
                for hchain in heavy_antibody_list:
                    for lchain in light_antibody_list:
                        if hchain is not None and lchain is not None:
                            if hchain.ori_name != lchain.ori_name:
                                paired_list.append(PairAntiBody(hchain, lchain, is_bsab=True))
            
            if paired_list == []:
                result_list = self.heavy_antibody + self.light_antibody
            else:
                result_list = paired_list.copy()
            
            return result_list


def seq_sliding_window(seq, window_size, step_size):
    for i in range(0, len(seq), step_size):
        yield seq[i : i + window_size]


def merge_fasta_files(fasta_files_path, merged_fasta_path):
    names_groups = []
    names_list = []
    seqs_list = []
    for fasta_file_path in fasta_files_path:
        names, seqs = read_fasta_file(fasta_file_path)
        names_groups.append([s.split("|")[0].lstrip(">") for s in names])
        names_list.extend(names)
        seqs_list.extend(seqs)

    with open(merged_fasta_path, "w") as f:
        f.write("\n".join(chain(*zip(names_list, seqs_list))))

    return names_groups


def merge_fasta_files_sliding(fasta_files_path, merged_fasta_path):
    names_groups = []
    names_list = []
    seqs_list = []
    for fasta_file_path in fasta_files_path:
        names = []
        seqs = []
        temp_names, temp_seqs = read_fasta_file(fasta_file_path)
        for name, seq in zip(temp_names, temp_seqs):
            if len(seq) > 200:
                slid_seqs = list(seq_sliding_window(seq, 150, 100))
                slid_names = [name for _ in range(len(slid_seqs))]
                seqs.extend(slid_seqs)
                names.extend(slid_names)
            else:
                seqs.append(seq)
                names.append(name)
        
        names_groups.append([s.split("|")[0].lstrip(">") for s in names])
        names_list.extend(names)
        seqs_list.extend(seqs)

    with open(merged_fasta_path, "w") as f:
        f.write("\n".join(chain(*zip(names_list, seqs_list))))

    return names_groups


def para_clonotype_results(clonotype_file_path):
    df = pd.read_csv(clonotype_file_path)
    clonotype = list(df["Clonotype"])
    seqs_names = list(df["Sequence_Name"])

    name_to_clonotype = {}
    for n, c in zip(seqs_names, clonotype):
        name = n.split("|")[0].lstrip(">")
        name_to_clonotype[name] = c

    # if len(name_to_clonotype) != len(clonotype):
    #     raise ValueError("The names of sequences are not unique.")

    return name_to_clonotype


def para_align_resutls(
    heavy_regions_path, light_regions_path, heavy_clone_path, light_clone_path, merged_fasta_path,
):
    ori_names, ori_seqs = read_fasta_file(merged_fasta_path)
    ori_names = [n.split("|")[0].lstrip(">") for n in ori_names]
    ori_names_to_seqs = {n: s for n, s in zip(ori_names, ori_seqs)}
    
    antibodes_heavy = []
    antibodes_light = []
    if os.path.exists(heavy_regions_path):
        names, seqs = read_fasta_file(heavy_regions_path)
        names = [n.split("|")[0].lstrip(">") for n in names]
        seqs = [s.replace("-", "") for s in seqs]
        # if len(set(names)) < len(names):
        #     raise ValueError("The names of sequences are not unique.")

        antibodes_heavy = [
            AntiBodySingle(name, seq, "H", full_seq=ori_names_to_seqs[name], ori_name=name) for name, seq in zip(names, seqs)
        ]
        name_to_clonotype_heavy = para_clonotype_results(heavy_clone_path)
        for antibody in antibodes_heavy:
            antibody.set_clonotype(name_to_clonotype_heavy[antibody.name])

    if os.path.exists(light_regions_path):
        names, seqs = read_fasta_file(light_regions_path)
        names = [n.split("|")[0].lstrip(">") for n in names]
        seqs = [s.replace("-", "") for s in seqs]
        # if len(set(names)) < len(names):
        #     raise ValueError("The names of sequences are not unique.")

        antibodes_light = [
            AntiBodySingle(name, seq, "L", full_seq=ori_names_to_seqs[name], ori_name=name) for name, seq in zip(names, seqs)
        ]
        name_to_clonotype_light = para_clonotype_results(light_clone_path)
        for antibody in antibodes_light:
            antibody.set_clonotype(name_to_clonotype_light[antibody.name])

    return antibodes_heavy, antibodes_light


def group_antibodies(antibody_heavy, antibody_light, names_groups, merged_names_to_seqs):
    for i, names in enumerate(names_groups):
        ab = AntiBody()
        abs_list = []
        for name in names:
            temp_ab_list = []
            for antibody in antibody_heavy + antibody_light:
                if antibody.name == name:
                    temp_ab_list.append(antibody)
                    ab.add_name(name, merged_names_to_seqs[name])
            
            abs_combinations = list(combinations(temp_ab_list, 2))
            for ab_combination in abs_combinations:
                if ab_combination[0].chain_type != ab_combination[1].chain_type:
                    ab.add_inner_pair_antibody(PairAntiBody(ab_combination[0], ab_combination[1]))
            
            abs_list.extend(temp_ab_list)

        if len(abs_list) > len(names):
            ab_names = [ab.name for ab in abs_list]
            # if len(set(ab_names)) != len(names):
            #     raise ValueError("The names of sequences are not unique.")
            # else:
            for sab in abs_list:
                sab.name = sab.name + "_" + str(merged_names_to_seqs[sab.name].find(sab.seq.replace("*","")))

        if len(abs_list) == 1:
            if abs_list[0].chain_type == "H":
                ab.add_heavy_antibody(abs_list[0])
            else:
                ab.add_light_antibody(abs_list[0])
        else:
            for antibody in abs_list:
                if antibody.chain_type == "H":
                    ab.add_heavy_antibody(antibody)
                else:
                    ab.add_light_antibody(antibody)

        yield ab


def get_chain_info(args):
    fasta_dir = args.fasta_dir
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)

    fasta_files_path = [
        os.path.join(fasta_dir, f)
        for f in os.listdir(fasta_dir)
        if f.endswith(".fasta") or f.endswith(".fas")
    ]

    merged_fasta_path = os.path.join(temp_dir, "all_sequences.fasta")
    slid_merged_fasta_path = os.path.join(temp_dir, "all_sequences_sliding.fasta")

    names_groups = merge_fasta_files(fasta_files_path, merged_fasta_path)
    slid_names_groups = merge_fasta_files_sliding(fasta_files_path, slid_merged_fasta_path)

    all_sequences_heavy_regions_path = os.path.join(
        temp_dir, "all_sequences_sliding_heavy.fas.temp.txt"
    )
    all_sequences_light_regions_path = os.path.join(
        temp_dir, "all_sequences_sliding_light.fas.temp.txt"
    )
    all_sequences_heavy_clone_path = os.path.join(
        temp_dir, "all_sequences_sliding_heavy.vgene.clonotype_seqs.csv"
    )
    all_sequences_light_clone_path = os.path.join(
        temp_dir, "all_sequences_sliding_light.vgene.clonotype_seqs.csv"
    )

    get_clonotype(
        fas_file=slid_merged_fasta_path,
        out_dir=temp_dir,
        chain_type="H",
        regioned_file=True,
        cpus=args.cpus,
    )

    get_clonotype(
        fas_file=slid_merged_fasta_path,
        out_dir=temp_dir,
        chain_type="L",
        regioned_file=True,
        cpus=args.cpus,
    )

    antibodes_heavy, antibodes_light = para_align_resutls(
        all_sequences_heavy_regions_path,
        all_sequences_light_regions_path,
        all_sequences_heavy_clone_path,
        all_sequences_light_clone_path,
        merged_fasta_path,
    )
    
    merged_names, merged_seqs = read_fasta_file(merged_fasta_path)
    merged_names_to_seqs = {n.lstrip(">"):s for n, s in zip(merged_names, merged_seqs)}
    
    antibodes_list = [
        ab for ab in group_antibodies(antibodes_heavy, antibodes_light, names_groups, merged_names_to_seqs)
    ]

    return antibodes_list
