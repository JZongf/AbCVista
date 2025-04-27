import os
from utils.align import run_alignment, delete_msa_by_first_seq
from utils.database import heavy_ab_database_path, light_ab_database_path
from utils.fasta import read_fasta_file, write_fasta_file


def merge_fasta_content(file_path_list, out_path, fasta_name=None, fasta_seq=None):
    names_list = []
    seqs_list = []
    for file_path in file_path_list:
        names, seqs = read_fasta_file(file_path)
        names_list.extend(names)
        seqs_list.extend(seqs)
    
    if fasta_seq is not None:
        if fasta_name is None:
            fasta_name = ">query"
        names_list = [fasta_name] + names_list
        seqs_list = [fasta_seq] + seqs_list
    write_fasta_file(names_list, seqs_list, out_path)


def get_msa(args):
    cpus = args.cpus
    fasta_dir = args.fasta_dir
    tmp_dir = args.temp_dir
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
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
    if not os.path.exists(out_alignments_dir):
        os.makedirs(out_alignments_dir)
    
    for file_path in fasta_path:
        base_name = os.path.basename(file_path).split(".")[0]
        seq_to_dict = {}
        original_names, original_seqs = read_fasta_file(file_path)
        for chain_id, name in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", original_names):
            seq_to_dict[name.lstrip(">").rstrip("_")] = chain_id
    
        for db_name, db_path in heavy_ab_database_path.items():
            heavy_temp_path = os.path.join(tmp_dir, "{}_heavy.fasta".format(os.path.basename(file_path).split(".")[0]))
            run_alignment(fas_file=file_path, 
                          out_path=heavy_temp_path, 
                          scheme="chothia", 
                          chain="H", 
                          get_regioned_file=False,
                          cpus=cpus
            )
            if os.path.exists(heavy_temp_path):
                names_list, seqs_list = read_fasta_file(heavy_temp_path)
                for index, name in enumerate(names_list):
                    name = name.lstrip(">")
                    name = name.rstrip("_")
                    if fold_type == "openfold":
                        result_msa_dir = "{}/alignments/{}".format(output_dir, name)
                    else:
                        result_msa_dir = "{}/{}/msas/{}".format(output_dir, base_name, seq_to_dict[name.strip("_")])
                    if not os.path.exists(result_msa_dir):
                        os.makedirs(result_msa_dir)
                        
                    result_msa_path = os.path.join(result_msa_dir, "{}_hits.a3m".format(db_name))
                    if use_precomputed_msas and os.path.exists(result_msa_path):
                        continue
                    merge_fasta_content([db_path], heavy_temp_path, fasta_name=">"+name, fasta_seq=seqs_list[index])
                    run_alignment(fas_file=heavy_temp_path,
                                    out_path=result_msa_path,
                                    scheme="chothia",
                                    chain="H",
                                    get_regioned_file=False,
                                    merge=True,
                                    cpus=cpus
                    )
                    msa_names_list, msa_seqs_list = delete_msa_by_first_seq(result_msa_path)
                    write_fasta_file(msa_names_list, msa_seqs_list, result_msa_path)
                    os.remove(heavy_temp_path)

        for db_name, db_path in light_ab_database_path.items():
            light_temp_path = os.path.join(tmp_dir, "{}_light.fasta".format(os.path.basename(file_path).split(".")[0]))
            run_alignment(fas_file=file_path, 
                          out_path=light_temp_path, 
                          scheme="chothia", 
                          chain="L", 
                          get_regioned_file=False,
                          cpus=cpus
            )
            if os.path.exists(light_temp_path):
                names_list, seqs_list = read_fasta_file(light_temp_path)
                for index, name in enumerate(names_list):
                    name = name.lstrip(">")
                    name = name.rstrip("_")
                    if fold_type == "openfold":
                        result_msa_dir = "{}/alignments/{}".format(output_dir, name)
                    else:
                        result_msa_dir = "{}/{}/msas/{}".format(output_dir, base_name, seq_to_dict[name.strip("_")])
                    
                    if not os.path.exists(result_msa_dir):
                        os.makedirs(result_msa_dir)
                    result_msa_path = os.path.join(result_msa_dir, "{}_hits.a3m".format(db_name))
                    if use_precomputed_msas and os.path.exists(result_msa_path):
                        continue
                    merge_fasta_content([db_path], light_temp_path, fasta_name=">"+name, fasta_seq=seqs_list[index])
                    run_alignment(fas_file=light_temp_path,
                                    out_path=result_msa_path,
                                    scheme="chothia",
                                    chain="L",
                                    get_regioned_file=False,
                                    merge=True,
                                    cpus=cpus
                    )
                    msa_names_list, msa_seqs_list = delete_msa_by_first_seq(result_msa_path)
                    write_fasta_file(msa_names_list, msa_seqs_list, result_msa_path)
                    os.remove(light_temp_path)