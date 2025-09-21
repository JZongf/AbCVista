import os
from utils import get_msa_by_clonotype
from utils import get_msa_by_single
from utils import get_msa_by_substitute
from utils import get_msa_by_pair
from utils.multiprocess import dynamic_executor_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.database import (
    unpaired_database_path, heavy_length_databases, 
    light_length_databases, clone_heavy_database_path, 
    clone_light_database_path, fv_length_database_path, 
    heavy_ab_database_path, light_ab_database_path
)


def get_msa(args, antibody_list):
    
    result_list = []
    length_heavy_max = 0
    length_light_max = 0
    ab_heavy_max = 0
    ab_light_max = 0
    databases_path = args.databases_path
    unpaired_database_length_dict = get_msa_by_single.get_database_length_dict(databases_path)
    fv_lengths_max, fv_lengths_q3, fv_database_path = get_msa_by_pair.get_database_stats(databases_path)

    with  dynamic_executor_context(process_threshold=1, max_workers=args.cpus) as dynamic_executor:
        futures = []
        executor = dynamic_executor.get_executor(10)
        for antibody in antibody_list:
            if antibody.is_paired() and (length_heavy_max == 0 or length_light_max == 0):
                length_heavy_max, length_light_max = get_msa_by_substitute.get_sub_max_length(databases_path, heavy_length_databases, light_length_databases)
            elif ab_heavy_max == 0 or ab_light_max == 0:
                ab_heavy_max, ab_light_max = get_msa_by_substitute.get_sub_max_length(databases_path, heavy_ab_database_path, light_ab_database_path)
            
            # Search sequences for unpaired MSA
            for heavy_chain in antibody.heavy_antibody:
                futures.append(executor.submit(get_msa_by_clonotype.build_msa, args, heavy_chain, "H"))
                futures.append(executor.submit(get_msa_by_single.build_msa, args, heavy_chain, "chothia", "H", unpaired_database_length_dict))
                if antibody.is_paired():
                    futures.append(executor.submit(get_msa_by_substitute.build_msa, args, heavy_chain, "chothia", "H", heavy_length_databases, length_heavy_max))
                else:
                    futures.append(executor.submit(get_msa_by_substitute.build_msa, args, heavy_chain, "chothia", "H", heavy_ab_database_path, ab_heavy_max))

            for light_chain in antibody.light_antibody:
                futures.append(executor.submit(get_msa_by_clonotype.build_msa, args, light_chain, "L"))
                futures.append(executor.submit(get_msa_by_single.build_msa, args, light_chain, "chothia", "L", unpaired_database_length_dict))
                if antibody.is_paired():
                    futures.append(executor.submit(get_msa_by_substitute.build_msa, args, light_chain, "chothia", "L", light_length_databases, length_light_max))
                else:
                    futures.append(executor.submit(get_msa_by_substitute.build_msa, args, light_chain, "chothia", "L", light_ab_database_path, ab_light_max))
            
            # Search sequences for paired MSA
            paired_antibodies = antibody.get_all_antibodies()
            if paired_antibodies == None:
                continue
            if len(paired_antibodies) == 1:
                paired_idx = None
            else:
                paired_idx = 0
            for pair in paired_antibodies:
                try:
                    if pair.is_paired():
                        futures.append(executor.submit(get_msa_by_pair.build_msa, args, pair, fv_lengths_max, fv_lengths_q3, fv_database_path, "paired_hits", paired_idx))
                        paired_idx += 1
                except Exception as e:
                    pass
            
            # Search sequences for inner-pair MSA
            for i, inner_pair in enumerate(antibody.inner_pair_antibody):
                if inner_pair != None:
                    futures.append(
                        executor.submit(
                            get_msa_by_pair.build_msa, args, inner_pair, fv_lengths_max, fv_lengths_q3, fv_database_path, f"inner_pair_hits_{i}"
                        )
                    )
            
        with ThreadPoolExecutor(max_workers=1) as realign_executor:
            paired_idx_start = 0
            for future in as_completed(futures):
                result_list = future.result()

                if result_list[0].get("out_temp_name_list"):
                    # Perform paired-sequence multi-sequence alignment
                    paired_idx_start += realign_executor.submit(
                        get_msa_by_pair.process_feature,
                        result_list,
                        args.cpus,
                    ).result()
                    if paired_idx_start > 100000000:
                        paired_idx_start = 0

                else:
                    # Perform unpaired-sequence multi-sequence alignment
                    for results in result_list:
                        if type(results) == dict and results.get("out_fasta_temp_path"):
                            realign_executor.submit(
                                get_msa_by_clonotype.realign_msa, 
                                results["out_fasta_temp_path"], 
                                results["out_msa_path"], 
                                "chothia", 
                                args.cpus, 
                                chain_type=results["chain_type"]
                            )
                