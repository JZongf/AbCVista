import torch
import numpy as np
import pandas as pd
from openfold.data.data_transforms import curry1
from openfold.data import msa_pairing
from openfold.data.cluster import get_cluster_by_embedding
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from collections import Counter
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from openfold.np.residue_constants import HHBLITS_AA_TO_ID
from openfold.np.residue_constants import ID_TO_HHBLITS_AA, OUR_ID_TO_HHBLITS_AA
from utils.get_antibody_region import REGION_NAME
from utils import blosum
import random
from typing import List, Union, Optional

REGION_WEIGHT_MATRIX_HEAVY = [0, 0, 1, 1, 1, 1, 3, 1, 0] # 重链的权重矩阵，由于测序原因FR1可能会缺失，所以FR1的权重为0
REGION_WEIGHT_MATRIX_LIGHT = [0, 0, 1, 1, 1, 1, 1, 1, 0] # 轻链的权重矩阵

REGION_WEIGHT_MATRIX_HEAVY_CLUSTER = [0, 0, 1, 1, 1, 1, 1, 1, 0] # 重链的权重矩阵，由于测序原因FR1可能会缺失，所以FR1的权重为0
REGION_WEIGHT_MATRIX_LIGHT_CLUSTER = [0, 0, 1, 1, 1, 1, 1, 1, 0] # 轻链的权重矩阵

MSA_CROP_SIZE = 10240
MSA_SINGLE_CROP_SIZE = 20480

def ab_hamming_distance(s1: Union[List[int], np.ndarray], 
                                 s2: Union[List[int], np.ndarray], 
                                 weight_matrix: Optional[Union[List[float], np.ndarray]] = None) -> float:
    """
    Vectorized Hamming distance calculation for integer sequences.
    
    Parameters:
    -----------
    s1, s2 : array-like
        Integer sequences to compare
    weight_matrix : array-like, optional
        Weights for each position
        
    Returns:
    --------
    float
        Weighted Hamming distance
    """
    # 确保输入是numpy数组
    s1_arr = np.asarray(s1, dtype=np.int32)
    s2_arr = np.asarray(s2, dtype=np.int32)
    
    if len(s1_arr) != len(s2_arr):
        return 0.0
    
    # 计算不相等的位置
    mask = (s1_arr != s2_arr)
    
    if weight_matrix is None:
        return float(np.sum(mask))
    else:
        weights = np.asarray(weight_matrix, dtype=np.float64)
        return float(np.sum(weights * mask))


def batch_hamming_distance_vectorized(sequences: Union[List[List[int]], np.ndarray], 
                                    reference: Union[List[int], np.ndarray], 
                                    weight_matrix: Optional[Union[List[float], np.ndarray]] = None) -> np.ndarray:
    """
    Vectorized batch computation of Hamming distances for multiple sequences.
    
    Parameters:
    -----------
    sequences : array-like
        2D array of sequences to compare
    reference : array-like
        Reference sequence to compare against
    weight_matrix : array-like, optional
        Weights for each position
        
    Returns:
    --------
    np.ndarray
        Array of distances
    """
    # 确保输入是numpy数组
    seq_matrix = np.asarray(sequences, dtype=np.int32)
    ref_arr = np.asarray(reference, dtype=np.int32)
    
    # 计算不相等的位置
    mask = (seq_matrix != ref_arr)
    
    if weight_matrix is None:
        return np.sum(mask, axis=1).astype(np.float64)
    else:
        weights = np.asarray(weight_matrix, dtype=np.float64)
        return np.sum(mask * weights, axis=1)


def select_representative_sequences(clusters, distances,n):
    # 计算label to index的dict
    label_to_index = {}
    for i, cluster in enumerate(clusters):
        if cluster not in label_to_index:
            label_to_index[cluster] = [i]
        else:
            label_to_index[cluster].append(i)
    
    selected_idx = []
    # 迭代所有聚类
    for cluster, indices in label_to_index.items():
        if len(indices) < n:
            selected_idx.extend(indices)
        else:
            temp_distance = np.array(distances)[indices]
            # 计算聚类中心
            center_distance = np.mean(temp_distance, axis=0)
            # 计算聚类中心与每个样本的距离
            temp_distance = temp_distance - center_distance
            temp_distance = np.sum(temp_distance, axis=1)
            # 选取距离最近的n个样本
            sorted_index = np.argsort(temp_distance)[:n]
            selected_idx.extend(np.array(indices)[sorted_index])
    
    selected_idx = sorted(selected_idx)
    return selected_idx


def rerange_and_crop_single_msa_by_chain(np_chains_list):
    """
    reorder the single MSA according to the consistency between the antibody sequence and the query sequence.
    And crop the MSA to the specified length.
    """
    for i, chain in enumerate(np_chains_list):
        # get the weight matrix
        region_index = chain["region_index"]
        weight_matrix = []

        if region_index["chain_type"] == "H":
            weight_base = REGION_WEIGHT_MATRIX_HEAVY
        else:
            weight_base = REGION_WEIGHT_MATRIX_LIGHT
            
        for i, weight in enumerate(weight_base):
            region = REGION_NAME[i]
            weight_matrix.extend([weight] * (region_index[region][1] - region_index[region][0]))
        
        # rerange the msa according to the weight matrix and the query sequence
        msa = np.array(chain["msa"])
        tg_seq = np.array(msa[0])
        scores_list = batch_hamming_distance_vectorized(msa, tg_seq, weight_matrix)

        min_crop_size = min(MSA_CROP_SIZE, msa.shape[0])
        for k in chain:
            k_split = k.split('_all_seq')[0]
            if k_split in msa_pairing.MSA_FEATURES and '_all_seq' not in k:
                sorted_item = sorted(zip(scores_list, chain[k]), key=lambda x: x[0])
                sorted_result = [item[1] for item in sorted_item]
                chain[k] = sorted_result[:min_crop_size]
    
    return np_chains_list


def crop_paired_msa(feature_dict):
    """
    Crop the MSAs to the specified length.
    """
    region_info = feature_dict["region_index"]
    chain_types = []
    for info in region_info:
        info = info[0]
        chain_types.append(info["chain_type"])
        
    msa = feature_dict["msa"]
    msa_mask = feature_dict["msa_mask"]
    deletion_matrix = feature_dict["deletion_matrix"]
    msas_num = feature_dict["msas_num"]
    original_msa_num = msas_num.copy()
    
    msas_num[0] = min(MSA_CROP_SIZE, msas_num[0])
    paired_msa = msa[:msas_num[0]]
    paired_msa_mask = msa_mask[:msas_num[0]]
    paired_deletion_matrix = deletion_matrix[:msas_num[0]]
    
    for i, chain_type in enumerate(chain_types, 1):
        if chain_type == "H":
            heavy_crop_size = min(MSA_CROP_SIZE, msas_num[i])
            msas_num[i] = heavy_crop_size
        else:
            light_crop_size = min(MSA_CROP_SIZE, msas_num[i])
            msas_num[i] = light_crop_size
    
    
    chain_msa_list = []
    chain_msa_mask = []
    chain_deletion_matrix = []
    start_index = 0
    for i, chain_num in enumerate(msas_num):
        chain_num += start_index
        chain_msa_list.append(msa[start_index:chain_num])
        chain_msa_mask.append(msa_mask[start_index:chain_num])
        chain_deletion_matrix.append(deletion_matrix[start_index:chain_num])
        start_index += original_msa_num[i]
    
    msa = np.concatenate(chain_msa_list)
    msa_mask = np.concatenate(chain_msa_mask)
    deletion_matrix = np.concatenate(chain_deletion_matrix)
    
    feature_dict["msa"] = msa
    feature_dict["msa_mask"] = msa_mask
    feature_dict["deletion_matrix"] = deletion_matrix
    feature_dict["single_msa_num"] = sum(msas_num[1:])
    feature_dict["chain1_msa_num"] = msas_num[1]
    feature_dict["chain2_msa_num"] = msas_num[2]
    feature_dict["pair_msa_num"] = msas_num[0]
    feature_dict["num_alignments"] = len(msa)
    
    return feature_dict


def crop_single_msa(feature_dict):
    crop_size = min(MSA_SINGLE_CROP_SIZE, len(feature_dict["msa"]))
    feature_dict["msa"] = feature_dict["msa"][:crop_size]
    feature_dict["num_alignments"] = len(feature_dict["msa"])
    feature_dict["msa_mask"] = feature_dict["msa_mask"][:feature_dict["num_alignments"]]
    feature_dict["deletion_matrix"] = feature_dict["deletion_matrix"][:feature_dict["num_alignments"]]
    
    return feature_dict


def rerange_msa_multimer(msa, pair_msa_num, chain1_msa_num, chain2_msa_num, region_index, msas_lenght):
    """
    For paired antibodies, reorder the MSA according to the consistency between the antibody sequence and the query sequence. 
    The paired chain, heavy chain, and light chain are sorted separately.
    """

    # 将msa分为3部分进行排序
    tg_seq = msa[0]
    temp_msa = []
    split_index = [0]

    weight_matrix = []
    for region_info in region_index:
        region_info = region_info[0]
        if region_info["chain_type"] == "H":
            weight_base = REGION_WEIGHT_MATRIX_HEAVY
            for i, weight in enumerate(weight_base):
                region = REGION_NAME[i]
                weight_matrix.extend([weight] * (region_info[region][1] - region_info[region][0]))
                
        elif region_info["chain_type"] == "L":
            weight_base = REGION_WEIGHT_MATRIX_LIGHT
            for i, weight in enumerate(weight_base):
                region = REGION_NAME[i]
                weight_matrix.extend([weight] * (region_info[region][1] - region_info[region][0]))
                
        else:
            weight_base = REGION_WEIGHT_MATRIX_HEAVY
            weight_matrix.extend([0] * region_info["length"])

    if sum(weight_matrix) == 0:
        weight_matrix = [1 for _ in range(len(tg_seq))]
    
    temp_lenght = 0
    for msa_l in msas_lenght:
        temp_lenght += msa_l
        split_index.append(temp_lenght)
    
    for start_index, end_point in enumerate(split_index[1:]):
        sub_msa_arrar =  np.array(msa[split_index[start_index]:end_point])
        ref = np.array(tg_seq)
        score_list = batch_hamming_distance_vectorized(sub_msa_arrar, ref, weight_matrix)
        
        sorted_item = sorted(zip(score_list, msa[split_index[start_index]:end_point]), key=lambda x: x[0])
        sorted_msa = [item[1] for item in sorted_item]
        temp_msa.extend(sorted_msa)
    
    return temp_msa


def rerange_msa(msa, region_index):
    """
    Reorder MSA based on the consistency between the antibody sequence and the query sequence
    """
    chain_type = region_index[0][0]["chain_type"]
    
    if chain_type == "H":
        weight_base = REGION_WEIGHT_MATRIX_HEAVY
    else:
        weight_base = REGION_WEIGHT_MATRIX_LIGHT
    
    weight_matrix = []
    for region_info in region_index:
        region_info = region_info[0]
        if region_info["chain_type"] == "H" or region_info["chain_type"] == "L":
            for i, weight in enumerate(weight_base):
                region = REGION_NAME[i]
                weight_matrix.extend([weight] * (region_info[region][1] - region_info[region][0]))
        else:
            weight_matrix.extend([1] * len(msa[0]))
    
    tg_seq = msa[0]
    tg_seq = np.array(tg_seq)
    msa = np.array(msa)
    score_list = batch_hamming_distance_vectorized(msa, tg_seq, weight_matrix)

    sorted_item = sorted(zip(score_list, msa), key=lambda x: x[0])
    sorted_msa = [item[1] for item in sorted_item]
    
    return sorted_msa


def generate_split_list(end_index, split_count, start_index=0):
    """
    Generate a list that splits max_len into split_count sublists
    """
    temp_list = list(range(start_index, end_index))
    list_len = end_index - start_index
    split_list = []
    for i in range(split_count):
        split_list.append(temp_list[i*list_len//split_count:(i+1)*list_len//split_count])
    
    return split_list


@curry1
def sample_msa2_multimer(
    batch, 
    max_seq,
    max_extra_msa_seq,
    iter_idx=0, 
    max_iter=20,
    region_index=None,
    region_mask=False,
    msas_num=[],
    random_sample=False,
):
    """
    batch: A dictionary of various features
    max_seq: Maximum sequence length
    max_extra_msa_seq: Maximum additional sequence length
    iter_idx: Iteration number
    max_iter: Maximum number of iterations in one sampling
    sample_iter_time: Number of small samples in one large sample
    """
    # 将msa分为3部分进行排序
    temp_msa = []
    split_index = [0]
    # if pair_msa_num != 0:
    #     split_index.append(pair_msa_num)
    # split_index.append(pair_msa_num + chain1_msa_num)
    # split_index.append(pair_msa_num + chain1_msa_num + chain2_msa_num)
    
    temp_num = 0
    for msa_num in msas_num:
        temp_num += msa_num
        split_index.append(temp_num)
    
    sel_idx = []
    extra_sel_idx = []
    current_sample_time = iter_idx
    max_sample_time = max(max_iter-1, 1)
    for start_index, end_point in enumerate(split_index[1:]):
        split_idx_list = generate_split_list(end_point, max_seq//(len(split_index)-1), start_index=split_index[start_index])
        for idx in split_idx_list:
            if len(idx) > 0:
                if random_sample:
                    sel_idx.append(random.choice(idx))
                else:
                    temp_sel_idx = (len(idx)//max_sample_time) * current_sample_time
                    temp_sel_idx = min(temp_sel_idx, len(idx)-1)
                    if current_sample_time >= len(idx):
                        sel_idx.append(idx[-1])
                    else:
                        sel_idx.append(idx[temp_sel_idx])
            
        # 获取选择的额外msa的索引
        extra_split_idx_list = generate_split_list(end_point, max_extra_msa_seq//(len(split_index)-1), start_index=split_index[start_index])
        
        extra_sample_time = current_sample_time
        # # 排除掉已经选择的序列
        # if max_seq == max_extra_msa_seq:
        #     extra_sample_time = current_sample_time + 1

        for idx in extra_split_idx_list:
            if len(idx) > 0:
                if random_sample:
                    extra_sel_idx.append(random.choice(idx))
                else:
                    temp_extra_sel_idx = (len(idx)//max_sample_time) * extra_sample_time
                    if max_seq == max_extra_msa_seq:
                        temp_extra_sel_idx += 1
                    temp_extra_sel_idx = min(temp_extra_sel_idx, len(idx)-1)
                    if current_sample_time >= len(idx):
                        extra_sel_idx.append(idx[-1])
                    else:
                        extra_sel_idx.append(idx[temp_extra_sel_idx])

    sel_idx[0] = 0
    extra_sel_idx[0] = 0
    for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
        if k in batch:
            batch['extra_' + k] = batch[k][extra_sel_idx]
            batch[k] = batch[k][sel_idx]
    
    # if region_index is not None and region_mask:
    #     # 如果需要mask的话，将CDR3区域的氨基酸序列设为0
    #     # 由于CDR3区域的长度不一样，所以需要根据region_index来进行mask
    #     # 每一轮sample_iter_time的迭代，都会mask掉一部分CDR3区域
    #     mask_msa_num = int((sample_iter_time - iter_idx%sample_iter_time + 1) / sample_iter_time * batch['msa'].shape[0])
    #     extra_mask_msa_num = int((sample_iter_time - iter_idx%sample_iter_time + 1) / sample_iter_time * batch['extra_msa'].shape[0])
    #     start_index = 0
    #     for index in region_index:
    #         if index[0]["chain_type"] == "H":
    #             batch['msa'][:mask_msa_num, start_index+index[0]["CDR3"][0]:start_index+index[0]["CDR3"][1]] = 0 # 将CDR3区域的氨基酸序列设为0
    #             batch["extra_msa"][:extra_mask_msa_num, start_index+index[0]["CDR3"][0]:start_index+index[0]["CDR3"][1]] = 0
    #             start_index += index[0]["length"]
    
    return batch


@curry1
def sample_msa2(
    batch, 
    max_seq,
    max_extra_msa_seq,
    iter_idx=0, 
    max_iter=20,
    region_index=None,
    region_mask=False,
):
    """
    batch: A dictionary of various features
    max_seq: Maximum sequence length
    max_extra_msa_seq: Maximum additional sequence length
    iter_idx: Iteration number
    max_iter: Maximum number of iterations in one sampling
    sample_iter_time: Number of small samples in one large sample
    """
    # 将msa分为3部分进行排序
    sel_idx = []
    extra_sel_idx = []
    current_sample_time = iter_idx
    max_seqs = len(batch["msa"])
    max_sample_time = max(max_iter-1, 1)

    split_idx_list = generate_split_list(max_seqs, max_seq, start_index=0)
    for idx in split_idx_list:
        temp_sel_idx = (len(idx)//max_sample_time) * current_sample_time
        temp_sel_idx = min(temp_sel_idx, len(idx)-1)
        if current_sample_time >= len(idx):
            sel_idx.append(idx[-1])
        else:
            sel_idx.append(idx[temp_sel_idx])
    
    extar_idx_list = generate_split_list(max_seqs, max_extra_msa_seq, start_index=0)
    for idx in extar_idx_list:
        temp_extra_sel_idx = (len(idx)//max_sample_time) * current_sample_time
        
        # # 排除掉已经选择的序列
        # if idx[temp_extra_sel_idx] in sel_idx:
        #     temp_extra_sel_idx += 1
            
        temp_extra_sel_idx = min(temp_extra_sel_idx, len(idx)-1)
        if current_sample_time >= len(idx):
            extra_sel_idx.append(idx[-1])
        else:
            extra_sel_idx.append(idx[temp_extra_sel_idx])

    sel_idx[0] = 0
    extra_sel_idx[0] = 0
    for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
        if k in batch:
            batch['extra_' + k] = batch[k][extra_sel_idx]
            batch[k] = batch[k][sel_idx]
    
    # if region_index is not None and region_mask:
    #     # 如果需要mask的话，将CDR3区域的氨基酸序列设为0
    #     # 由于CDR3区域的长度不一样，所以需要根据region_index来进行mask
    #     # 每一轮sample_iter_time的迭代，都会mask掉一部分CDR3区域
    #     mask_msa_num = int((sample_iter_time - iter_idx%sample_iter_time + 1) / sample_iter_time * batch['msa'].shape[0])
    #     extra_mask_msa_num = int((sample_iter_time - iter_idx%sample_iter_time + 1) / sample_iter_time * batch['extra_msa'].shape[0])
    #     start_index = 0
    #     for index in region_index:
    #         if index[0]["chain_type"] == "H":
    #             batch['msa'][:mask_msa_num, start_index+index[0]["CDR3"][0]:start_index+index[0]["CDR3"][1]] = 0 # 将CDR3区域的氨基酸序列设为0
    #             batch["extra_msa"][:extra_mask_msa_num, start_index+index[0]["CDR3"][0]:start_index+index[0]["CDR3"][1]] = 0
    #             start_index += index[0]["length"]
    
    
    return batch


def filter_labels(paired_msa_labels, cutoff=16):
    # 计算每个label对的频度
    pair_counts = Counter(paired_msa_labels)

    # 使用filter函数排除掉频度小于16的label对
    filtered_labels = [pair for pair in paired_msa_labels if pair_counts[pair] >= cutoff]

    return filtered_labels


def get_best_eps(distances, k):
    # 使用NearestNeighbors类计算每个点的k-距离
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(distances)
    # 获取k个最近邻的距离
    k_distances, indices = neigh.kneighbors(n_neighbors=k)
    # 对k-距离进行排序
    sorted_distances = np.sort(k_distances[:, k-1], axis=0)

    # 使用KneeLocator找到膝点
    knee_locator = KneeLocator(
        range(len(sorted_distances)), 
        sorted_distances, 
        curve='convex', 
        direction='increasing', 
        online=True
    )
    eps = knee_locator.all_knees_y[-2] # 最左边和最右边的膝点无效，所以取倒数第二个
    
    return eps


def msa_cluster(
    msa, 
    region_index, 
    cluster_params={},
    max_msa_clusters=64,
    embedding_batch_size=256,
):
    """
    Cluster the MSA into several groups according to the distance between the antibody sequence and the query sequence.
    """
    assert cluster_params["method"] in ["kmeans", "hdbscan"]

    msa_aa = ["".join([OUR_ID_TO_HHBLITS_AA[aa_id] for aa_id in seq]) for seq in msa]
    selected_clusters, ori_clusters = get_cluster_by_embedding(
        msa = msa_aa,
        regions = [[region_index[0][0]["CDR3"][0], region_index[0][0]["CDR3"][1]]],
        cluster_params=cluster_params,
        batch_size=256,
        with_pair=False,
    )

    cluster_label = selected_clusters
    cluster_label_set = set(cluster_label)
    cluster_idx = {}
    
    # group the index by the label
    for idx, label in enumerate(cluster_label):
        if label in cluster_label_set:
            if label not in cluster_idx:
                cluster_idx[label] = []
            cluster_idx[label].append(idx)
    
    # filter the cluster index by the number of sequences in each cluster
    cluster_idx = list(cluster_idx.values())
    # cluster_idx = [sub_list for sub_list in cluster_idx if len(sub_list) > max_msa_clusters//2]

    return cluster_idx


def split_list(lst, n):
    """
    Split the list into n sublists
    """
    length = len(lst)
    return [lst[i*length//n: (i+1)*length//n] for i in range(n)]


def msa_cluster_multimer(
    msa,
    region_index, 
    msas_num=[],
    cluster_params={},
    max_msa_clusters=64,
    embedding_batch_size=256,
):
    assert cluster_params["method"] in ["kmeans", "hdbscan"]
    paired_msa_labels = []
    
    # get each msa's sequence size
    split_index = []
    start_index = 0
    for i, msa_num in enumerate(msas_num):
        split_index.append((start_index, start_index+msa_num))
        start_index += msa_num
    
    # get each chain's length
    chain_lens = []
    for region_info in region_index:
        region_info = region_info[0]
        chain_lens.append(region_info["length"])

    # rebuild the msa, including the paired msa and the single msa
    paired_msa = msa[split_index[0][0]:split_index[0][1]]
    extend_msa = []
    len_start = 0
    for x in range(1, len(msas_num)):
        single_msa = msa[split_index[x][0]:split_index[x][1]]
        len_end = chain_lens[x-1]+len_start
        extend_msa.append(
            list(np.array(paired_msa)[:, len_start:len_end]) + 
            list(np.array(single_msa)[:, len_start:len_end])
        )
        len_start += chain_lens[x-1]
    
    chains_labels_list = []
    for index, temp_msa in enumerate(extend_msa):
        region_info = region_index[index][0]
        
        # id for msa to aa
        temp_msa_aa = ["".join([OUR_ID_TO_HHBLITS_AA[aa_id] for aa_id in seq]) for seq in temp_msa]
        temp_msa_len = len(temp_msa_aa[0])
        if region_info["length"] != temp_msa_len:
            raise ValueError("The length of the paired msa and the single msa is not equal!")
        
        # get the region for clustering
        if region_index[index][0]["chain_type"] == "H":
            region_cdr3 = (region_index[index][0]["CDR3"][0]-3, region_index[index][0]["CDR3"][1])
            regions = [region_cdr3]
        elif region_index[index][0]["chain_type"] == "L":
            region_cdr3 = (region_index[index][0]["CDR3"][0], region_index[index][0]["CDR3"][1])
            regions = [region_cdr3]
        else:
            regions = [(0, region_index[index][0]["length"])]

        # cluster the paired msa
        selected_single_clusters, pair_clusters = get_cluster_by_embedding(
            msa = temp_msa_aa,
            regions = regions,
            cluster_params=cluster_params,
            batch_size=256,
            with_pair=True,
            pair_size=len(paired_msa),
        )
        paired_msa_labels.append(pair_clusters)
        chains_labels_list.append(selected_single_clusters)

    msa_lens = msas_num.copy()
    paired_msa_labels = list(zip(*paired_msa_labels))
    set_msa_labels = set(paired_msa_labels)
    set_msa_labels = set(pair for pair in set_msa_labels if -1 not in pair)
    cluster_labels = []
    paired_seq_cutoff = 3
    while cluster_labels == []:
        cluster_labels = []
        for paired_labe in set_msa_labels:
            
            temp_cluster_index = [] # all the sequence index in the same cluster
            temp_seq_count = 0 # sequence count for a cluster
            temp_pair_index = []
            for idx, pl in enumerate(paired_msa_labels): # get paired sequence index
                if pl == paired_labe:
                    temp_pair_index.append(idx)
                    temp_seq_count += 1
            if len(temp_pair_index) < paired_seq_cutoff:
                continue
            temp_cluster_index.append(temp_pair_index)
            
            for chain_idx, chain_labels in enumerate(chains_labels_list): # get single sequence index
                temp_single_index = [] 
                for seq_idx, cl in enumerate(chain_labels):
                    if cl == paired_labe[chain_idx]:
                        temp_single_index.append(sum(msa_lens[:chain_idx+1]) + seq_idx)
                        temp_seq_count += 1
                if len(temp_single_index) < 1:
                    continue
                temp_cluster_index.append(temp_single_index)
            
            if temp_seq_count > 15:
                cluster_labels.append(temp_cluster_index)
    
    return cluster_labels


def sample_fixed_msa(
    cluster_idx_list,
    max_seq
):
    """
    Fixed the number of sequences in each cluster, and sample the MSA.
    
    cluster_idx_list: list of cluster index list
    max_seq: the maximum number of sequences in each cluster
    return: the index list of the sampled MSA
    """
    cluster_idx_to_len_list = [[cluster, len(cluster)] for cluster in cluster_idx_list]
    cluster_idx_to_len_list = sorted(cluster_idx_to_len_list, key=lambda x: x[1], reverse=False)
    
    result_idx_list = []
    sampled_count = 0
    for i, (cluster_idx, cluster_len) in enumerate(cluster_idx_to_len_list):
        temp_sample_count = (max_seq - sampled_count) // (len(cluster_idx_to_len_list) - i)
        temp_sample_count = min(temp_sample_count, cluster_len)
        sampled_count += temp_sample_count
        
        splited_idx_list = split_list(cluster_idx, temp_sample_count)
        for idxs in splited_idx_list:
            idx = len(idxs) // 2
            result_idx_list.append(idxs[idx])
    
    return result_idx_list


@curry1
def sample_msa_cluster(
    batch, 
    max_seq,
    max_extra_msa_seq,
    iter_idx=0, 
    region_index=None,
    region_mask=False,
    msa_cluster_idx=None,
    fix_cluster_size=False,
):
    """
    Sample the MSA according to the clustering results.
    """
    current_sample_time = iter_idx
    cluster_idx = msa_cluster_idx[current_sample_time]
    
    if isinstance(cluster_idx[0], list):
        for i in range(len(cluster_idx)):
            cluster_idx[i] = cluster_idx[i][:512]
        if fix_cluster_size:
            cluster_idx = sample_fixed_msa(cluster_idx, max_seq)
        else:
            cluster_idx = sum(cluster_idx, [])
    else:
        cluster_idx = cluster_idx[:512]

    sel_idx = cluster_idx
    sel_idx[0] = 0
    extra_sel_idx = sel_idx.copy()
    for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
        if k in batch:
            batch['extra_' + k] = batch[k][extra_sel_idx]
            batch[k] = batch[k][sel_idx]
    
    return batch