import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
python_dir = os.path.dirname(sys.executable)

from datetime import date
from utils import align

align.init_Abalign_path(path=os.path.join(script_dir, "lib"))

from utils import (
    get_msa_by_clonotype,
    get_msa_by_length,
    get_msa_by_pair,
    get_msa_by_single,
    get_msa_by_substitute,
    msa_supplement,
    get_antibody_region,
    fasta,
    database,
    save_msa,
    get_chain_info,
    get_msa,
)
from scripts import search_template

import argparse
import logging
import math
import numpy as np
np.random.seed(42)

from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    run_model,
    prep_output,
    update_timings,
    relax_protein,
)

from openfold.data import sample_msa
from openfold.utils import structure_align

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle

import random
import time
import torch
import multiprocessing

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data.tools import hhsearch, hmmsearch
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.data import (
    templates,
    feature_pipeline,
    data_pipeline,
    feature_processing_multimer,
)
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from scripts import seqlogo
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import copy
from peft import LoraModel, PeftModel

TRACING_INTERVAL = 50


def clear_nested_list(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            clear_nested_list(item)
            item = None
            del item
        elif isinstance(item, torch.Tensor):
            item = None
            del item
        elif isinstance(item, np.ndarray):
            item = None
            del item
        elif isinstance(item, dict):
            clear_nested_dict(item)
            item = None
            del item
    nested_list = None

    del nested_list

def clear_nested_dict(nested_dict):
    for key, value in nested_dict.items():
        if isinstance(value, (list, tuple)):
            clear_nested_list(value)
        elif isinstance(value, torch.Tensor):
            del value
        elif isinstance(value, np.ndarray):
            del value
        elif isinstance(value, dict):
            clear_nested_dict(value)
    del nested_dict


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def replace_invalid_chars(name, replace_char=""):
    # Characters to Avoid in Pathnames
    invalid_chars = [
        "<",
        ":",
        '"',
        "/",
        "\\",
        "|",
        "?",
        "*",
        "+",
        "-",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    ]

    for char in invalid_chars:
        name = name.replace(char, replace_char)

    return name


def rewrite_seqs(input_dir, temp_dir):
    """
    Write the sequences to the temp_dir.
    args:
        input_dir: the input fasta dir
        alignment_dir: the alignment dir
        temp_dir: the temp dir
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    fasta_path_list = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".fasta") or f.endswith(".fa") or f.endswith(".fas")
    ]
    name_dict = {}
    for fasta_path in fasta_path_list:
        base_name = os.path.basename(fasta_path)
        names, seqs = fasta.read_fasta_file(fasta_path)
        names = [replace_invalid_chars(name) for name in names]

        for i, name in enumerate(names):
            name = name.replace(" ", "_")
            name = name.replace(",", "_")
            name = name.replace("\r", "_")
            name = name.replace("\n", "_")
            if len(name) == 2:
                name += "_" + base_name.split(".")[0]
            if name in name_dict:
                name += "_" + base_name.split(".")[0]
            name_dict[name] = True
            names[i] = name

        out_path = os.path.join(temp_dir, base_name)
        fasta.write_fasta_file(names, seqs, out_path)


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    if len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        # local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=alignment_dir
        )
    elif "multimer" in args.config_preset:
        with open(tmp_fasta_path, "w") as fp:
            fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=alignment_dir,
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path,
            super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def save_seqlogo_thread(
    tag, index, rank_list, dist_rank_list, used_msa, seqlogo_output_dir
):
    try:
        seqlogo_name = f"{tag}_ranked{rank_list[index]}_dranked{dist_rank_list[index]}"
        if not os.path.exists(seqlogo_output_dir):
            os.makedirs(seqlogo_output_dir)
        seqlogo.save_seqlogo(
            name=seqlogo_name, output_dir=seqlogo_output_dir, msa=used_msa
        )
    except Exception as e:
        print(f"Save seqlogo failed: {e}")


def relax_protein_thread(
    args,
    config,
    index,
    rank_list,
    drank_list,
    output_name,
    output_directory,
    unrelaxed_protein,
):
    try:
        relax_protein(
            config,
            args.model_device,
            unrelaxed_protein,
            output_directory,
            output_name
            + "_ranked{}_dranked{}".format(rank_list[index], drank_list[index]),
            args.cif_output,
        )
    except Exception as e:
        print(f"Relaxation failed: {e}")


def interface(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = model_config(
        args.config_preset, 
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        )

    # update config
    config.data.predict.max_msa_clusters = args.max_msa_clusters
    config.data.predict.max_extra_msa = args.max_extra_msa
    config.model.evoformer_stack.msa_dropout = args.msa_dropout
    config.model.evoformer_stack.pair_dropout = args.pair_dropout
    config.data.common.max_recycling_iters = args.sample_count
    config.data.common.sample_iter_time = args.sample_iter_time
    config.data.common.fix_cluster_size = args.fix_cluster_size
    config.model.early_stop = args.early_stop
    # set early stop to True if kmeans or hdbscan clustering is used
    if args.kmeans_cluster or args.hdbscan_cluster:
        config.model.early_stop = True
        
    if args.trace_model:
        if not config.data.predict.fixed_size:
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )

    is_multimer = "multimer" in args.config_preset

    if is_multimer:
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=args.hmmsearch_binary_path,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            database_path=args.pdb_seqres_database_path,
        )

        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )
    else:
        template_searcher = hhsearch.HHSearch(
            binary_path=args.hhsearch_binary_path,
            databases=[args.pdb70_database_path],
        )

        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    if is_multimer:
        data_processor = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=data_processor,
        )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    tag_list = []
    seq_list = []
    for fasta_file in list_files_with_extensions(
        args.fasta_dir, (".fasta", ".fa", ".fas")
    ):
        # Gather input sequences
        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)

        if (not is_multimer) and len(tags) != 1:
            print(
                f"{fasta_path} contains more than one sequence but "
                f"multimer mode is not enabled. Skipping..."
            )
            continue

        # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
        tag = "-".join(tags)

        tag_list.append((tag, tags))
        seq_list.append(seqs)

    seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
    sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
    model_generator = load_models_from_command_line(
        config,
        args.model_device,
        args.openfold_checkpoint_path,
        args.jax_param_path,
        args.output_dir,
        args = args,
    )
    
    for model, output_directory in model_generator:
        if torch.cuda.is_bf16_supported():
            model.to(dtype=torch.bfloat16)

        cur_tracing_interval = 0
        for (tag, tags), seqs in sorted_targets:
            output_name = (
                f"{tag[:50]}_{args.config_preset}"  # avoid too long output name
            )

            # skip exist output
            if args.skip_exist_output:
                exist_flag = False
                exist_files = os.listdir(output_directory)
                for exist_file in exist_files:
                    if exist_file.startswith(output_name):
                        exist_flag = True
                        break
                if exist_flag:
                    continue

            print(f"Predicting for {tag}...")
            process_feature_start_time = time.time()
            if args.output_postfix is not None:
                output_name = f"{output_name}_{args.output_postfix}"

            feature_dict = generate_feature_dict(
                tags,
                seqs,
                alignment_dir,
                data_processor,
                args,
            )
            
            with_other_domain = False # Check if there are regions other than the antibody variable region
            for region_index in feature_dict["region_index"]:
                if region_index[0]["chain_type"] == "O":
                    with_other_domain = True
                    break

            if args.trace_model:
                n = feature_dict["aatype"].shape[-2]
                rounded_seqlen = round_up_seqlen(n)
                feature_dict = pad_feature_dict_seq(
                    feature_dict,
                    rounded_seqlen,
                )

            # reranke msa
            if feature_dict.get("pair_msa_num", None) is not None:
                feature_dict["msa"] = sample_msa.rerange_msa_multimer(
                    feature_dict["msa"],
                    feature_dict["pair_msa_num"],
                    feature_dict["chain1_msa_num"],
                    feature_dict["chain2_msa_num"],
                    region_index=feature_dict["region_index"],
                    msas_lenght=feature_dict["msas_num"],
                )
            else:
                feature_dict["msa"] = sample_msa.rerange_msa(
                    feature_dict["msa"], region_index=feature_dict["region_index"]
                )

            if args.kmeans_cluster or args.hdbscan_cluster:
                if feature_dict.get("pair_msa_num", None) is not None:
                    
                    feature_dict = sample_msa.crop_paired_msa(feature_dict)
                    
                    if args.kmeans_cluster:
                        cluster_params = {
                            "method" : "kmeans",
                            "n_clusters" : args.kmeans_n_clusters_multimer,
                        }
                    elif args.hdbscan_cluster:
                        cluster_params = {
                            "method" : "hdbscan",
                            "min_cluster_size" : args.hdbscan_min_cluster_size_multimer,
                        }

                    feature_dict["msa_cluster_idx"] = (
                        sample_msa.msa_cluster_multimer(
                            feature_dict["msa"],
                            feature_dict["region_index"],
                            msas_num=feature_dict["msas_num"],
                            cluster_params=cluster_params,
                            max_msa_clusters=max(args.max_msa_clusters, args.max_extra_msa),
                            embedding_batch_size=args.embedding_batch_size,
                        )
                    )
                    
                else:
                    
                    feature_dict = sample_msa.crop_single_msa(feature_dict)
                    
                    if args.kmeans_cluster:
                        cluster_params = {
                            "method" : "kmeans",
                            "n_clusters" : args.kmeans_n_clusters,
                        }
                        
                    elif args.hdbscan_cluster:
                        cluster_params = {
                            "method" : "hdbscan",
                            "min_cluster_size" : args.hdbscan_min_cluster_size,
                        }

                    feature_dict["msa_cluster_idx"] = sample_msa.msa_cluster(
                        feature_dict["msa"],
                        feature_dict["region_index"],
                        cluster_params=cluster_params,
                        max_msa_clusters=max(args.max_msa_clusters, args.max_extra_msa),
                        embedding_batch_size=args.embedding_batch_size,
                    )

                # 记录个batch真实使用的msa数量
                # msa_length_list = [len(feature_dict["msa_cluster_idx"][i]) for i in range(len(feature_dict["msa_cluster_idx"]))]
                args.sample_count = len(feature_dict["msa_cluster_idx"])
                config.data.common.max_recycling_iters = args.sample_count
                msa_length_list = [
                    args.max_msa_clusters
                    for i in range(config.data.common.max_recycling_iters)
                ]
                max_msa_length = max(msa_length_list)
                # extra_msa_length_list = msa_length_list
                extra_msa_length_list = [
                    args.max_extra_msa
                    for i in range(config.data.common.max_recycling_iters)
                ]
                max_extra_msa_length = max_msa_length
            else:
                msa_length_list = [
                    args.max_msa_clusters
                    for i in range(config.data.common.max_recycling_iters)
                ]
                max_msa_length = args.max_msa_clusters
                extra_msa_length_list = [
                    args.max_extra_msa
                    for i in range(config.data.common.max_recycling_iters)
                ]
                max_extra_msa_length = args.max_extra_msa

            # # update max_msa_clusters and max_extra_msa in config
            # args.max_msa_clusters = max_msa_length
            # args.max_extra_msa = max_extra_msa_length
            # config.data.predict.max_msa_clusters = args.max_msa_clusters
            # config.data.predict.max_extra_msa = args.max_extra_msa

            config.data.predict.max_templates = feature_dict["template_all_atom_mask"].shape[0]
            config.data.predict.max_template_hits = feature_dict["template_all_atom_mask"].shape[0]
            feature_processor = feature_pipeline.FeaturePipeline(config.data)
            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode="predict", is_multimer=is_multimer
            )

            # Add the number of MSA clusters to the feature dict
            processed_feature_dict["msa_length"] = torch.as_tensor(
                np.array(
                    [
                        [length]
                        for length in msa_length_list
                        for i in range(processed_feature_dict["msa"].shape[-1])
                    ]
                ).T
            )
            processed_feature_dict["extra_msa_length"] = torch.as_tensor(
                np.array(
                    [
                        [length]
                        for length in extra_msa_length_list
                        for i in range(processed_feature_dict["msa"].shape[-1])
                    ]
                ).T
            )

            if args.trace_model:
                if rounded_seqlen > cur_tracing_interval:
                    logger.info(f"Tracing model at {rounded_seqlen} residues...")
                    t = time.perf_counter()
                    trace_model_(model, processed_feature_dict)
                    tracing_time = time.perf_counter() - t
                    logger.info(
                        "# Tracing finished, time costs {}s: {}, {}".format(
                            tracing_time, tag
                        )
                    )
                    cur_tracing_interval = rounded_seqlen
            
            process_feature_stop_time = time.time()
            logger.info("Feature generation time: {}s".format(process_feature_stop_time - process_feature_start_time))
            
            # get the original msa used by the model
            original_msa = processed_feature_dict["msa"].detach().clone().numpy()
            original_msa = np.stack(
                [
                    feature_processing_multimer._restore_msa_restypes(
                        original_msa[..., i].copy()
                    )
                    for i in range(original_msa.shape[-1])
                ],
                axis=-1,
            )  # resume custom restypes to HHBLITS restypes

            # Run the model
            if isinstance(model, PeftModel) and with_other_domain:
                with model.disable_adapter():
                    outputs_list = run_model(model, copy.deepcopy(processed_feature_dict), tag, args.output_dir)
            else:
                outputs_list = run_model(model, copy.deepcopy(processed_feature_dict), tag, args.output_dir)
            # outputs_list = structure_align.filter_outputs(outputs_list)
            
            # Toss out the recycling dimensions --- we don't need them anymore
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()), processed_feature_dict
            )

            # get hcdr3 lpddt if possible else get lpddt
            hcdr3_start = 0
            hcdr3_end = 0
            hcdr1_start = 0
            hcdr1_end = 0
            start_index = 0
            plddt_list = []
            for region in feature_dict["region_index"]:
                region = region[0]
                if region["chain_type"] == "H":
                    hcdr3_start = start_index + region["CDR3"][0]
                    hcdr3_end = start_index + region["CDR3"][1]
                    hcdr1_start = start_index + region["CDR1"][0]
                    hcdr1_end = start_index + region["CDR1"][1]

                if region["chain_type"] == "H" or region["chain_type"] == "L":
                    start_index += region["FR4"][1]
                else:
                    start_index += region["length"]

            if hcdr3_start != 0 and hcdr3_end != 0 and hcdr3_end - hcdr3_start > 0:
                for index, out in enumerate(outputs_list):
                    if hcdr1_end - hcdr1_start > 12:
                        cdr3_mean = out["plddt"][hcdr3_start:hcdr3_end].mean()
                        cdr1_mean = out["plddt"][hcdr1_start:hcdr1_end].mean()
                        plddt_list.append((cdr1_mean + cdr3_mean)/2)
                        
                    else:
                        plddt_list.append(out["plddt"][hcdr3_start:hcdr3_end].mean())
                    
            else:
                for index, out in enumerate(outputs_list):
                    plddt_list.append(out["plddt"].mean())

            # get rank by plddt or weighted iptm score
            if args.rank_by_ptm == False:
                # get rank by plddt
                rank_list = []
                for i, plddt in enumerate(plddt_list):
                    rank = 0
                    for j, plddt_ in enumerate(plddt_list):
                        if plddt_ > plddt:
                            rank += 1
                    rank_list.append(rank)
            else:
                # get weighted iptm score
                wiptm_list = []
                for i, wiptm in enumerate(outputs_list):
                    wiptm_list.append(outputs_list[i]["weighted_ptm_score"])

                rank_list = []
                for i, wiptm in enumerate(wiptm_list):
                    rank = 0
                    for j, wiptm_ in enumerate(wiptm_list):
                        if wiptm_ > wiptm:
                            rank += 1
                    rank_list.append(rank)

            # get the distance between Hcdr3s and center of Hcdr3s
            pos_hcdr3_list = []
            dist_rank_list = []
            for index, out in enumerate(outputs_list):
                if hcdr3_start != 0 and hcdr3_end != 0 and hcdr3_end - hcdr3_start > 0:
                    pos_hcdr3_list.append(
                        out["final_atom_positions"][hcdr3_start:hcdr3_end, :, :]
                    )
                else:
                    pos_hcdr3_list.append(
                        out["final_atom_positions"]
                    )
            dist_mean_list = structure_align.get_distance_to_center(pos_hcdr3_list)

            for i, dist_mean in enumerate(dist_mean_list):
                rank = 0
                for j, dist_mean_ in enumerate(dist_mean_list):
                    if dist_mean_ < dist_mean:
                        rank += 1
                dist_rank_list.append(rank)

            # Write the output
            for index, out in enumerate(outputs_list):

                unrelaxed_protein = prep_output(
                    out,
                    processed_feature_dict,
                    feature_dict,
                    feature_processor,
                    args.config_preset,
                    args.multimer_ri_gap,
                    args.subtract_plddt,
                )

                unrelaxed_file_suffix = "_unrelaxed_ranked{}_dranked{}.pdb".format(
                    rank_list[index], dist_rank_list[index]
                )
                if args.cif_output:
                    unrelaxed_file_suffix = "_unrelaxed_ranked{}_dranked{}.cif".format(
                        rank_list[index], dist_rank_list[index]
                    )
                unrelaxed_output_path = os.path.join(
                    output_directory, f"{output_name}{unrelaxed_file_suffix}"
                )

                with open(unrelaxed_output_path, "w") as fp:
                    if args.cif_output:
                        fp.write(protein.to_modelcif(unrelaxed_protein))
                    else:
                        fp.write(protein.to_pdb(unrelaxed_protein))

                logger.info(f"Output written to {unrelaxed_output_path}...")

                # save msas
                if args.save_used_msas:
                    umsa_output_dir = os.path.join(args.output_dir, "used_msas", tag)
                    if not os.path.exists(umsa_output_dir):
                        os.makedirs(umsa_output_dir)
                    used_msa_name = f"{tag}_ranked{rank_list[index]}_dranked{dist_rank_list[index]}.fas"
                    used_msa = original_msa[..., index]
                    save_msa.save_msa(
                        used_msa, os.path.join(umsa_output_dir, used_msa_name)
                    )

                # save output pkl
                if args.save_pkl:
                    features_output_dir = os.path.join(args.output_dir, "features")
                    if not os.path.exists(features_output_dir):
                        os.makedirs(features_output_dir)
                    output_dict_path = os.path.join(
                        features_output_dir,
                        "{}_output_dict_ranked{}_dranked{}.pkl".format(
                            output_name, rank_list[index], dist_rank_list[index]
                        ),
                    )
                    out["aatype"] = processed_feature_dict["aatype"]
                    select_features = ["asym_id", "plddt", "distogram_logits"]
                    select_out = {k: out[k] for k in select_features}
                    with open(output_dict_path, "wb") as fp:
                        pickle.dump(select_out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    logger.info(f"Model output written to {output_dict_path}...")

                # relax the prediction
                if not args.skip_relaxation and int(rank_list[index]) < 5:
                    # Relax the prediction.
                    logger.info(f"Running relaxation on {unrelaxed_output_path}...")
                    try:
                        relax_protein(
                            config,
                            args.model_device,
                            unrelaxed_protein,
                            output_directory,
                            output_name
                            + "_ranked{}_dranked{}".format(
                                rank_list[index], dist_rank_list[index]
                            ),
                            args.cif_output,
                        )
                    except Exception as e:
                        logger.info(f"Relaxation failed: {e}")
                        continue

            if args.save_seqlogo:
                with ProcessPoolExecutor(max_workers=args.cpus) as executor:
                    seqlogo_output_dir = os.path.join(args.output_dir, "seqlogo", tag)
                    if not os.path.exists(seqlogo_output_dir):
                        os.makedirs(seqlogo_output_dir)

                    for index in range(original_msa.shape[-1]):
                        used_msa = original_msa[..., index].copy()
                        executor.submit(
                            save_seqlogo_thread,
                            tag,
                            index,
                            rank_list,
                            dist_rank_list,
                            used_msa,
                            seqlogo_output_dir,
                        )

            ## clean up
            clear_nested_dict(feature_dict)
            clear_nested_dict(processed_feature_dict)
            clear_nested_list(outputs_list)
            original_msa = None
            del feature_dict, processed_feature_dict, outputs_list, original_msa
            gc.collect()
            torch.cuda.empty_cache()


def main(args):
    # get alignments dir
    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(args.output_dir, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    if not os.path.exists(alignment_dir):
        os.makedirs(alignment_dir)

    args.temp_dir = os.path.join(args.temp_dir, str(os.getpid()))

    # write antibody sequence to temp dir
    abseq_temp_dir = os.path.join(os.path.join(args.temp_dir, "input_fasta"))
    rewrite_seqs(args.fasta_dir, abseq_temp_dir)
    args.fasta_dir = abseq_temp_dir

    if args.use_precomputed_alignments is None:
        # build alignments
        time_start = time.time()

        # get antibody specific msas
        batch_size = 500 # Limit the thread pool size to prevent tasks from getting stuck
        antibody_list_full = get_chain_info.get_chain_info(args=args)
        for batch_i in range(0, len(antibody_list_full), batch_size):
            antibody_list = antibody_list_full[batch_i:batch_i+batch_size]
            # get msa
            database.init_databases_path(database_path=args.databases_path)
            get_msa.get_msa(args=args, antibody_list=antibody_list) # all
            
            # get antibody region file
            get_antibody_region.write_regions(alignment_dir, antibody_list, args=args) 

            # padding the antibody msas
            msa_supplement.padding_msas(args)

            # TODO: 构建inner_paired_msa和outer_paired_msa
            msa_supplement.merge_msas(args, antibody_list, alignment_dir)

            # search template
            search_template.search(alignment_dir, antibody_list, args)

        time_end = time.time()
        print("\n* Build alignments time: {}s\n".format(time_end - time_start))

    # # run interface
    interface(args)

    if os.path.exists(args.temp_dir):
        try:
            os.removedirs(args.temp_dir)
        except Exception as e:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_dir",
        type=str,
        help="Path to directory containing FASTA files, the file extension should be fa, fas or fasta",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="""Name of the directory in which to output the results""",
    )
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored.""",
    )
    parser.add_argument(
        "--use_precomputed_msas",
        action="store_true",
        default=False,
        help="Whether to skip the msas that has already been generated",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda:0",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")""",
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        default="model_1_multimer_v3",
        help="""Name of a model config preset defined in openfold/config.py""",
    )
    parser.add_argument(
        "--jax_param_path",
        type=str,
        default=None,
        help="""Path to JAX model parameters of Alphafold2. """,
    ) 
    parser.add_argument(
        "--openfold_checkpoint_path",
        type=str,
        default=os.path.join(
            script_dir, "database/params_model_1_multimer_v3_lora.pt",
        ),
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file""",
    )
    parser.add_argument(
        "--long_sequence_inference",
        action="store_true",
        default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details""",
    )
    parser.add_argument(
        "--use_deepspeed_evoformer_attention", action="store_true", default=False, 
        help="Whether to use the DeepSpeed evoformer attention layer. Must have deepspeed installed in the environment.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="""Number of CPUs to use for running Abalign. 
        Default is half of the available CPUs.""",
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="""Postfix for output prediction filenames""",
    )
    parser.add_argument("--data_random_seed", type=str, default=None)
    parser.add_argument(
        "--skip_relaxation",
        action="store_true",
        default=False,
        help="If True, the relaxation process is skipped, otherwise the first five structures will be relaxed.",
    )
    parser.add_argument(
        "--multimer_ri_gap",
        type=int,
        default=400,
        help="""Residue index offset between multiple sequences, if provided""",
    )
    parser.add_argument(
        "--trace_model",
        action="store_true",
        default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs.""",
    )
    parser.add_argument(
        "--subtract_plddt",
        action="store_true",
        default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself""",
    )
    parser.add_argument(
        "--cif_output",
        action="store_true",
        default=False,
        help="Output predicted models in ModelCIF format instead of PDB format (default)",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="/tmp/AbCFold",
        help="Path to directory containing temporary files.",
    )
    parser.add_argument(
        "--fold_type",
        type=str,
        default="openfold",
        help="The type used for structural prediction",
    )
    parser.add_argument(
        "--skip_exist_output",
        action="store_true",
        default=False,
        help="Skip the output file that already exists.",
    )

    # template config
    parser.add_argument(
        "--hmmsearch_binary_path",
        type=str,
        default=os.path.join(python_dir, "hmmsearch"),
    )
    parser.add_argument(
        "--hmmbuild_binary_path", type=str, default=os.path.join(python_dir, "hmmbuild")
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default=os.path.join(python_dir, "kalign")
    )
    parser.add_argument(
        "--template_mmcif_dir",
        type=str,
        default=os.path.join(
            script_dir, "database/mmcif_files_ab"
        ),
    )
    parser.add_argument(
        "--max_template_date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--obsolete_pdbs_path",
        type=str,
        default=os.path.join(script_dir, "database/obsolete.dat"),
    )
    parser.add_argument(
        "--pdb_seqres_database_path",
        type=str,
        default=os.path.join(
            script_dir, "database/pdb_seqres_ab.txt"
        ),
    )

    # custom_config
    parser.add_argument(
        "--max_msa_clusters",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max_extra_msa",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--msa_dropout",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--pair_dropout",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--databases_path", type=str, default=os.path.join(script_dir, "database")
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=20,
        help="""The number of structures to sample, 
        controlling how many structures are predicted for each sequence. 
        This parameter is ignored if clustering methods are used.
        """
    )
    parser.add_argument(
        "--sample_iter_time",
        type=int,
        default=4,
        help="The number of recycling iterations for each sampling.",
    )
    parser.add_argument(
        "--rank_by_ptm",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_pkl",
        action="store_true",
        default=False,
        help="Whether to save the pkl file of model output.",
    )
    parser.add_argument(
        "--save_used_msas",
        action="store_true",
        default=False,
        help="Whether to save the used msas.",
    )
    parser.add_argument(
        "--save_seqlogo",
        action="store_true",
        default=False,
        help="Whether to save the seqlogo.",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        default=False,
        help="Whether to enable early stop for sampling.",
    )
    
    # # cluster params
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=256,
        help="""The batch size for sequence embeddings 
        when calculating cosine similarity and obtaining clustering results. 
        Reducing this value can decrease GPU memory usage, 
        but will increase the embedding time.""",
    )
    parser.add_argument(
        "--fix_cluster_size",
        action="store_true",
        default=True,
        help="""Whether to fix the cluster size to equal number.""",
    )
    # kmeans cluster params
    parser.add_argument(
        "--kmeans_cluster",
        action="store_true",
        default=False,
        help="""Whether to enable the k-means algorithm for clustering.
        K-means and HDbscan cannot be enabled simultaneously.""",
    )
    parser.add_argument(
        "--kmeans_n_clusters",
        type=int,
        default=20,
        help="The cluster count for kmeans cluster.",
    )
    parser.add_argument(
        "--kmeans_n_clusters_multimer",
        type=int,
        default=20,
        help="""The cluster count for kmeans cluster for multimer.
        The final number of clusters will be larger than this value 
        due to the pairing of light and heavy chain clusters.
        """,
    )
    
    # hdbscan cluster params
    parser.add_argument(
        "--hdbscan_cluster",
        action="store_true",
        default=False,
        help="""Whether to enable the HDBSCAN algorithm for clustering.
        K-means and HDBSCAN cannot be enabled simultaneously.""",
        
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size",
        type=int,
        default=8,
        help="The min cluster size for hdbscan cluster.",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size_multimer",
        type=int,
        default=8,
        help="The min cluster size for hdbscan cluster for multimer.",
    )

    args = parser.parse_args()

    main(args)
