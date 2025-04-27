import os
import json
from time import time
from . import hmmsearch
from concurrent.futures import ThreadPoolExecutor


def search_template(inputfile, template_searcher, use_precomputed_msas):
    out_dir = os.path.dirname(inputfile)
    if use_precomputed_msas and os.path.exists(os.path.join(out_dir, "hmm_output.sto")):
        return

    if os.path.exists(inputfile):
        uniref90_msa_f = open(inputfile, "r")
        uniref90_msa_as_a3m = uniref90_msa_f.read()

        pdb_templates_result = template_searcher.query(
            uniref90_msa_as_a3m,
            output_dir=out_dir,
        )


def search(data_dir, antibody_list, args):

    search_path_list = []
    for ab in antibody_list:
        is_paired = ab.is_paired()
        for name, seq in ab.names_to_seqs.items():
            
            flag = False # If true, the MSA is used as input for the template search, otherwise the sequence itself is used as input
            dir_path = os.path.join(data_dir, name)
            region_path = os.path.join(dir_path, "region_index.json")
            msa_path = os.path.join(dir_path, "uniref90_hits.a3m")
            
            if os.path.exists(region_path) and is_paired:
                with open(region_path, "r") as rf:
                    regions = json.load(rf)
                    if (regions["FRONT"][1] - regions["FRONT"][0] > 0) or (regions["BACK"][1] - regions["BACK"][0] > 0):
                        flag = False # False if there are other areas besides the variable domain
                    else:
                        flag = True
            
            if flag:
                search_path_list.append(msa_path)

            else:
                with open(msa_path, "r") as f:
                    temp_lines = f.readlines()[:2]
                temp_path = os.path.join(os.path.dirname(msa_path), "temp.fasta")
                with open(temp_path, "w") as f:
                    f.writelines(temp_lines)
                search_path_list.append(temp_path)

    searcher = hmmsearch.Hmmsearch(
        binary_path=args.hmmsearch_binary_path,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        database_path=args.pdb_seqres_database_path,
    )
    
    max_workers = min(args.cpus // 8, 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file in search_path_list:
            future = executor.submit(search_template, file, searcher, args.use_precomputed_msas)
