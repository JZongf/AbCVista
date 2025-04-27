from antiberty import AntiBERTyRunner
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import time
from collections import defaultdict
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN, KMeans

def read_fasta(fasta_file):
    """
    Read a fasta file and return a list
    """
    with open(fasta_file, 'r') as f:
        lines = f.read().splitlines()
        seqs = lines[1::2]
        names = lines[::2]
    
    return seqs, names


def read_region_fasta(fasta_file, spliter='*', region_list=None):
    """
    Read a fasta file that splits sequences by regions and return a list
    """
    
    with open(fasta_file, 'r') as f:
        lines = f.read().splitlines()
        seqs = lines[1::2]
        names = lines[::2]
    
    seqs_region = [seq.split(spliter) for seq in seqs]
    
    if isinstance(region_list, list):
        seqs_region = ["".join([region for i, region in enumerate(seq_region) if i in region_list]) for seq_region in seqs_region]
    
    return seqs_region, names
    

def ab_hamming_distance(s1, s2, weight_matrix=None):
    """
    Calculate the Hamming distance between two antibody sequences, 
    the sequences will be weighted according to the weight_matrix
    """
    if len(s1) != len(s2):
        return
    
    if weight_matrix is None:
        weight_matrix = np.ones(len(s1))
        
    assert len(weight_matrix) == len(s1)
    
    distance = 0
    for index, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            distance += weight_matrix[index]
    
    distance /= len(s1)
    return distance


def get_hamming_distance_matrix(msa, weight_matrix=None):
    """
    Calculate the Hamming distance matrix between all pairs of sequences in the MSA,
    the sequences will be weighted according to the weight_matrix
    """

    msa_size = len(msa)
    distance_matrix = np.zeros((msa_size, msa_size))
    for i in range(msa_size):
        for j in range(i+1, msa_size):
            distance_matrix[i,j] = ab_hamming_distance(msa[i], msa[j], weight_matrix)
            distance_matrix[j,i] = distance_matrix[i,j]
            
    return distance_matrix


def compute_hamming_matrix_torch(msa, weight_matrix=None):
    """
    计算MSA中所有序列之间的Hamming距离矩阵，使用CUDA进行加速
    """
    msa_size = len(msa)
    seq_length = msa[0].size(0)
    
    # 将输入序列堆叠成一个二维张量
    msa_tensor = torch.stack(msa).cuda()
    
    if weight_matrix is None:
        weight_matrix = torch.ones(seq_length, device=msa_tensor.device)
    
    # 计算每一对序列之间的Hamming距离
    # 使用广播机制来计算Hamming距离
    # 将 msa_tensor 转换为 (msa_size, seq_length) 和 (seq_length, msa_size) 进行比较
    hamming_diffs = msa_tensor.unsqueeze(1) != msa_tensor.unsqueeze(0)
    
    # 加权并计算距离
    distance_matrix = hamming_diffs.float().sum(dim=2) @ weight_matrix.unsqueeze(1)
    
    # 返回对称的距离矩阵
    return distance_matrix


def get_best_eps(distances, k=8, knee_index=-1):
    """
    使用 k-距离图和 KneeLocator 算法自动寻找最佳 eps 值。

    Args:
        distances (np.ndarray): 距离矩阵或数据点坐标。
        k (int): k-近邻的数量。默认为 5。
        knee_index (int): 要使用的膝点索引. 默认为 -2 (倒数第二个).  
                           可以使用 0 获取最显著的膝点.

    Returns:
        float: 最佳 eps 值。
    """
    if distances.ndim == 2 and distances.shape[0] == distances.shape[1]:  # 距离矩阵
        k_distances = np.sort(distances, axis=1)[:, :k]
        sorted_distances = np.sort(k_distances[:, k-1])
    else: # 数据点坐标
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(distances)
        k_distances, _ = neigh.kneighbors(distances)
        sorted_distances = np.sort(k_distances[:, k-1])


    knee_locator = KneeLocator(
        range(len(sorted_distances)),
        sorted_distances,
        curve='convex',
        direction='increasing',
    )

    if knee_locator.all_knees_y:
        try:
            eps = knee_locator.all_knees_y[knee_index]
        except IndexError:
            print("Warning: Invalid knee_index. Returning the most prominent knee.")
            eps = knee_locator.knee  # 回退到最显著的膝点
    else:
        print("Warning: No knee found. Returning a default eps value.")
        eps = np.median(sorted_distances) # 或其他默认值

    return eps


def select_representative_sequences(distance_matrix, cluster_labels, n, start=0):
    # 将序列按聚类分组
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    
    representatives = {}
    for label, cluster in clusters.items():
        if len(cluster) <= n:
            # 如果聚类中的序列数量不超过n，全部选择
            representatives[label] = sorted(cluster)  # 排序以确保选择靠前的索引
        else:
            # 计算聚类中心点（使用到其他所有点的平均距离最小的点）
            cluster_distances = distance_matrix[np.ix_(cluster, cluster)]
            center_scores = np.mean(cluster_distances, axis=1)
            
            # 在距离相等的情况下，优先选择索引较小的点作为中心
            center_idx = min(enumerate(center_scores), key=lambda x: (x[1], cluster[x[0]]))[0]
            center = cluster[center_idx]
            
            # 选择距离中心最近的n个点（包括中心点自身）
            distances_to_center = distance_matrix[center, cluster]
            
            # 创建一个复合键für排序：(距离, 原始索引)
            sorted_indices = sorted(enumerate(distances_to_center), key=lambda x: (x[1], cluster[x[0]]))
            closest_indices = [cluster[idx] for idx, _ in sorted_indices[:n]]
            
            representatives[label] = closest_indices

        representatives[label] = [i+start for i in representatives[label]]
    
    return representatives


def compute_cosine_similarity_torch(embeddings):
    # 假设 embeddings 是一个形状为 [n_seq, n_char, n_dim] 的 PyTorch tensor
    n_seq, n_char, n_dim = embeddings.shape
    
    # 如果可用，使用 GPU
    if torch.cuda.is_available():
        embeddings = embeddings.cuda()
    
    # 重塑 embeddings 为 [n_seq, n_char * n_dim]
    reshaped_embeddings = embeddings.reshape(n_seq, -1)
    
    # 计算范数
    norms = torch.norm(reshaped_embeddings, p=2, dim=1, keepdim=True)
    
    # 归一化嵌入
    normalized_embeddings = reshaped_embeddings / norms
    
    # 直接计算余弦相似性矩阵
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    return similarity_matrix


def cosinner_similarity_cluster(
    vectors, 
    cluster_params={}, 
    with_pair=False, 
    pair_size=0,
):
    # calculate cosine similarity matrix
    similarity_matrix = compute_cosine_similarity_torch(vectors)
    similarity_matrix = similarity_matrix.cpu().numpy()
    
    # convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_matrix
    
    # set negative values to zero
    distance_matrix[distance_matrix < 0] = 0
    
    if cluster_params["method"] == "hdbscan":
        clustering = HDBSCAN(min_cluster_size=cluster_params["min_cluster_size"], min_samples=cluster_params["min_cluster_size"], metric='precomputed', n_jobs=-1)
    elif cluster_params["method"] == "kmeans":
        clustering = KMeans(n_clusters=cluster_params["n_clusters"], random_state=0)
        
    # get cluster labels
    clustering.fit(distance_matrix)
    labels = clustering.labels_.tolist()
    
    # get the distance matrix of the pair and single sequences
    if with_pair:
        pair_distance_matrix = distance_matrix[np.ix_(range(pair_size), range(pair_size))]
        pair_labels = clustering.labels_[:pair_size]
        
        pair_labels_to_index = {}
        for i, label in enumerate(pair_labels):
            if label not in pair_labels_to_index:
                pair_labels_to_index[label] = [i]
            else:
                pair_labels_to_index[label].append(i)
        
        single_distance_matrix = distance_matrix[np.ix_(range(pair_size, len(labels)), range(pair_size, len(labels)))]
        single_labels = clustering.labels_[pair_size:]
        selected_idx_single = select_representative_sequences(
            single_distance_matrix, 
            single_labels, 
            n=128,
            start=pair_size,
        )
        
        return selected_idx_single, pair_labels_to_index
    
    label_to_index = {}
    for i, label in enumerate(labels):
        if label not in label_to_index:
            label_to_index[label] = [i]
        else:
            label_to_index[label].append(i)
    
    selected_idx = select_representative_sequences(distance_matrix, labels, n=128)
    
    return selected_idx, label_to_index
    

def get_cluster_by_embedding(
    msa: list[str],
    regions: tuple[int, int],
    cluster_params={},
    batch_size=256,
    with_pair=False,
    pair_size=0,
):
    embedding_start = time.time()
    antiberty_runner = AntiBERTyRunner()
    
    total_seqs = len(msa)
    seq_len = len(msa[0])
    # 预先分配一个大的tensor来存储所有结果
    embeddings = torch.empty((total_seqs, sum([region[1]-region[0] for region in regions]), 512), dtype=torch.float32, device='cuda')
    
    with torch.no_grad():
        for i in range(0, total_seqs, batch_size):
            sub_seqs = msa[i:min(i+batch_size, total_seqs)]
            sub_embeddings = torch.stack(antiberty_runner.embed(sub_seqs))
            
            # 直接将结果复制到预分配的tensor中
            end_idx = min(i+batch_size, total_seqs)
            # embeddings[i:end_idx] = sub_embeddings[:, region[0]+1:region[1]+1, :]
            idx_list = sum([list(range(region[0], region[1])) for region in regions], [])
            embeddings[i:end_idx] = sub_embeddings[:, idx_list, :]

            # 确保当前批次的计算结果已同步到GPU
            torch.cuda.synchronize()
            
            # 清理不再需要的临时变量
            del sub_embeddings
            torch.cuda.empty_cache()

    embedding_end = time.time()
    print("Embedding time:", embedding_end - embedding_start)

    clustering_start = time.time()
    
    selected_clusters, ori_clusters = cosinner_similarity_cluster(
        embeddings,
        cluster_params=cluster_params,
        with_pair=with_pair,
        pair_size=pair_size,
    )
    # 构建一个长度为total_seqs的列表，每个元素为-1，表示未被分配到任何聚类
    selected_cluster_labels = [-1] * total_seqs
    for i, cluster in selected_clusters.items():
        if cluster_params["method"] == "hdbscan":
            i = i + 1  # 因为HDBSCAN的标签会有-1，所以这里需要加1
        for idx in cluster:
            selected_cluster_labels[idx] = i
        
    if with_pair:
        selected_cluster_labels = selected_cluster_labels[pair_size:]
    
    ori_cluster_labels = [-1] * total_seqs
    for i, cluster in ori_clusters.items():
        if cluster_params["method"] == "hdbscan":
            i = i + 1  # 因为HDBSCAN的标签会有-1，所以这里需要加1
        for idx in cluster:
            ori_cluster_labels[idx] = i
            
    if with_pair:
        ori_cluster_labels = ori_cluster_labels[:pair_size]

    clustering_end = time.time()
    print("Clustering time:", clustering_end - clustering_start)

    # 找出cluster中大聚类的数量
    cluster_sizes = sorted([len(cluster) for cluster in selected_clusters.values()], reverse=True)
    max_cluster_size = cluster_sizes[0]
    print("Max cluster size:", max_cluster_size)
    print("Number of clusters:", len(selected_clusters))
    
    return selected_cluster_labels, ori_cluster_labels

