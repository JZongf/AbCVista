import os
import io
import numpy as np
from Bio import PDB
from Bio.PDB import vectors, PDBIO
from subprocess import Popen, PIPE
import random
import time

# 获取当前文件夹的路径
current_path = os.path.abspath(os.path.dirname(__file__))
AbCFold_path = os.path.split(os.path.split(current_path)[0])[0]
FIXER_PATH = os.path.join(AbCFold_path, "lib/fixer/Fixer")
CISRR_PATH = os.path.join(AbCFold_path, "lib/CISRR/bin/CISRR")


def generate_seqres_records(seqs, missing_residues):
    PDB.Polypeptide.one_to_three
    seqs_1to3 = { chain_id:[PDB.Polypeptide.one_to_three(aa) for aa in seq] for chain_id, seq in seqs}
    
    remark = """
REMARK   1   
REMARK   1   MISSING RESIDUES
REMARK   1   THE FOLLOWING RESIDUES WERE NOT LOCATED IN THE
REMARK   1   EXPERIMENT. (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN
REMARK   1   IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)
REMARK   1   
REMARK   1   M RES C SSSEQI
"""
    for chain_id, missing_residues in missing_residues.items():
        for (residue_id, residue_name), (residue_id_2, residue_name_2) in missing_residues:
            remark += f"REMARK   1     {residue_name} {chain_id:<4}   {residue_id[1]:>4}   \n"
    
    seqres = ""
    for i, (chain_id, seq) in enumerate(seqs_1to3.items()):
        seq_length = len(seq)
        # 每一行只写13个残基
        count = 1
        for j in range(0, len(seq), 13):
            seqres += f"SEQRES{count:>4} {chain_id:>1} {seq_length:>4}  {' '.join(seq[j:j+13]):<60}\n"
            count += 1
        
    return remark + seqres


class AmideBondFixer:
    def __init__(self, pdb_string, output_dir="/tmp/", fixer_path=None, cisrr_path=None):
        """
        初始化AmideBondFixer类。
        
        参数:
            pdb_string (str): PDB文件内容。
            output_dir (str): 输出目录。
            fixer_path (str): Fixer工具路径。
            cisrr_path (str): CISRR工具路径。
        """
        self.pdb_string = io.StringIO(pdb_string)
        self.base_name = f"{random.randint(100000, 999999)}"
        self.output_dir = output_dir
        self.fixer_path = fixer_path or FIXER_PATH
        self.cisrr_path = cisrr_path or CISRR_PATH
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.residues_to_delete_dict = {}
        self.file_paths = {
            "broken_pdb": os.path.join(self.output_dir, f"{self.base_name}_broken.pdb"),
            "full_seqs": os.path.join(self.output_dir, f"{self.base_name}_full.fasta"),
            "fixed_pdb": os.path.join(self.output_dir, f"{self.base_name}_fixed.pdb"),
            "cisrr_pdb": os.path.join(self.output_dir, f"{self.base_name}_cisrr.pdb"),
            "filled_pdb": os.path.join(self.output_dir, f"{self.base_name}_filled.pdb"),
        }
    
    def calculate_omega_and_clean(self, protein, verbose=False):
        """
        计算PDB文件中的酰胺键ω角，删除非平面性残基（ω角偏离180°超过30°）。
        
        参数:
            protein (str): PDB文件内容。
            verbose (bool): 是否输出详细信息。
        
        返回:
            dict: 包含删除残基后的PDB文件路径、完整残基序列文件路径和需要删除的残基信息。
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", protein)
        self.ori_structure = structure.copy()
        
        deleted_residues_ids = []
        seqs = {}
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                residues = list(chain.get_residues())
                residues_to_delete = []
                seq = "".join(PDB.Polypeptide.three_to_one(r.get_resname()) for r in residues)
                seqs[chain_id] = seq
                
                for i in range(1, len(residues)):
                    prev_res = residues[i - 1]
                    curr_res = residues[i]
                    
                    try:
                        CA_prev = prev_res['CA'].get_vector()
                        C_prev = prev_res['C'].get_vector()
                        N_curr = curr_res['N'].get_vector()
                        CA_curr = curr_res['CA'].get_vector()
                        
                        angle = np.mod(vectors.calc_dihedral(CA_prev, C_prev, N_curr, CA_curr), 2 * np.pi)
                        
                        if 5 * np.pi / 6 <= angle <= 7 * np.pi / 6:
                            continue
                        
                        if (0 <= angle <= np.pi / 6) or (11 * np.pi / 6 <= angle <= 2 * np.pi):
                            if curr_res.resname == "PRO":
                                continue
                            if verbose:
                                print(f"CIS detected: Chain {chain_id}, Residues {prev_res.get_id()[1]}-{curr_res.get_id()[1]}, ω = {angle:.2f}°")
                        
                        residues_to_delete.append(((prev_res.get_id(), prev_res.get_resname()), (curr_res.get_id(), curr_res.get_resname())))
                        deleted_residues_ids.extend([i, i + 1])
                        if verbose:
                            print(f"Non-planar detected: Chain {chain_id}, Residues {prev_res.get_id()[1]}-{curr_res.get_id()[1]}, ω = {angle:.2f}°")
                    except KeyError:
                        continue
                
                if residues_to_delete:
                    self.residues_to_delete_dict[chain_id] = residues_to_delete
                
                for (prev_res_id, _), (curr_res_id, _) in residues_to_delete:
                    try:
                        chain.detach_child(prev_res_id)
                        if verbose:
                            print(f"Deleted chain: {chain_id}, residue: {prev_res_id}")
                    except Exception as e:
                        print(f"Error deleting residue: {prev_res_id}, {e}")
        
        if self.residues_to_delete_dict:
            pio = PDBIO()
            pio.set_structure(structure)
            pio.save(self.file_paths["broken_pdb"])
            
            with open(self.file_paths["full_seqs"], "w") as f:
                for chain_id, seq in seqs.items():
                    f.write(f">{chain_id}\n{seq}\n")
            
            return {
                "fixed_pdb_file": self.file_paths["broken_pdb"],
                "fixed_fasta_file": self.file_paths["full_seqs"],
                "residues_to_delete_dict": self.residues_to_delete_dict,
                "deleted_residues_ids": list(set(deleted_residues_ids)),
            }
        
        return {}
    
    def fixer(self, pdb_file, fasta_file):
        """
        使用Fixer工具修复PDB文件。
        
        参数:
            pdb_file (str): 输入PDB文件路径。
            fasta_file (str): 输入FASTA文件路径。
        
        返回:
            str: 修复后的PDB文件路径。
        """
        fix_cmd = f"{self.fixer_path} -i {pdb_file} -o {self.file_paths['fixed_pdb']} -s {fasta_file}"
        p = Popen(fix_cmd, shell=True, stdout=PIPE, stderr=PIPE)
        info, err = p.communicate()
        
        if err or "Error" in info.decode('utf-8'):
            print(f"Error fixing {pdb_file}: {err.decode('utf-8')}")
            return None
        
        return self.file_paths["fixed_pdb"]
    
    def cisrr(self, pdb_file, package_info):
        """
        使用CISRR工具处理PDB文件。
        
        参数:
            pdb_file (str): 输入PDB文件路径。
            package_info (dict): 需要删除的残基信息。
        
        返回:
            str: 处理后的PDB文件路径。
        """
        full_cmd = f"{self.cisrr_path} -i {pdb_file} -o {self.file_paths['cisrr_pdb']}"
        for chain_id, deleted_res_info in package_info.items():
            for (prev_res_id, prev_res_name), _ in deleted_res_info:
                seq_num = prev_res_id[1]
                full_cmd += f" -m {chain_id} {seq_num} {prev_res_name} {prev_res_name}"
        
        p = Popen(full_cmd, shell=True, stdout=PIPE, stderr=PIPE)
        _, err = p.communicate()
        
        if err:
            print(f"Error running CISRR on {pdb_file}: {err.decode('utf-8')}")
            return None
        
        return self.file_paths["cisrr_pdb"]
    
    def fill_bfactor_occupancy(self):
        """
        填充PDB文件中的B因子（B-factor）和占有率（occupancy）。
        """
        tmp_src = PDB.PDBParser(QUIET=True).get_structure("protein", self.file_paths["cisrr_pdb"])
        for model in tmp_src:
            for chain in model:
                for residue in chain:
                    ori_chain = self.ori_structure[model.id][chain.id]
                    ori_residue = ori_chain[residue.id]
                    for atom in residue:
                        try:
                            ori_atom = ori_residue[atom.id]
                            atom.set_occupancy(ori_atom.get_occupancy())
                            atom.set_bfactor(ori_atom.get_bfactor())
                        except KeyError:
                            print(f"Error filling atom: {atom.id}")
        
        io = PDB.PDBIO()
        io.set_structure(tmp_src)
        io.save(self.file_paths["filled_pdb"])
    
    def process(self):
        """
        处理PDB文件，修复酰胺键并填充B因子。
        
        返回:
            str: 处理后的PDB文件内容。
        """
        info = self.calculate_omega_and_clean(self.pdb_string, verbose=True)
        
        if self.residues_to_delete_dict:
            print("Fixing PDB file...")
            self.fixer(self.file_paths["broken_pdb"], self.file_paths["full_seqs"])
            self.cisrr(self.file_paths["fixed_pdb"], self.residues_to_delete_dict)
            self.fill_bfactor_occupancy()
            print("Done.")
            
            for file_type in ["fixed_pdb", "broken_pdb", "full_seqs", "cisrr_pdb"]:
                if os.path.exists(self.file_paths[file_type]):
                    os.remove(self.file_paths[file_type])
        else:
            return None, None
        
        with open(self.file_paths["filled_pdb"], "r") as f:
            pdb_string = f.read()
        
        if os.path.exists(self.file_paths["filled_pdb"]):
            os.remove(self.file_paths["filled_pdb"])
        
        return pdb_string, info
