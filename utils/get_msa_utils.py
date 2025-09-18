import pandas as pd
import numpy as np
from utils.database import regions, regions_fv
from utils.fasta import read_fasta_file,  save_data_to_pickle
from functools import lru_cache


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
    """在长度容差内筛选匹配的序列（保持结果与原实现一致）。

    - 输入可为 DataFrame，也可为等形状的 numpy 数组；
    - 返回类型与原实现一致：当输入为 DataFrame 时返回 DataFrame，并保持原行顺序；
      这能保证后续调用 `.to_numpy().tolist()` 的行为与顺序不变。
    - 实现改为基于 NumPy 的向量化计算，减少 Pandas 广播开销，加速大数据量筛选。
    """
    # 将容差转换为 numpy 向量，保证逐列比较完全等价
    if isinstance(tolerance, (list, tuple, np.ndarray)):
        tol_vec = np.asarray(tolerance)
    else:
        # 标量容差：与原行为一致（对所有列使用同一容差）
        # 需要按列广播，因此生成与列数一致的向量
        if hasattr(target_df, "shape"):
            col_n = target_df.shape[1]
        else:
            col_n = len(regions)  # 回退值，正常不会走到
        tol_vec = np.full((col_n,), tolerance)

    # 统一获取底层 ndarray，用于向量化计算（保持原始顺序）
    if isinstance(length_df, pd.DataFrame):
        length_arr = length_df.values
    else:
        length_arr = np.asarray(length_df)

    if isinstance(target_df, pd.DataFrame):
        target_arr = target_df.values
    else:
        target_arr = np.asarray(target_df)

    # 计算按列绝对差并与容差比较（逐列）
    # 形状：N x C，与原逻辑 (diff <= tolerance).all(axis=1) 等价
    diff = np.abs(length_arr - target_arr)
    mask = np.all(diff <= tol_vec, axis=1)

    # 返回与原实现一致的类型
    if isinstance(seqs_df, pd.DataFrame):
        return seqs_df[mask]
    else:
        # 非 DataFrame 情况下，返回 ndarray 的视图，保持顺序
        return np.asarray(seqs_df)[mask]


def filter_by_diff(
    seqs_df,
    diff_arr,
    tolerance=0,
):
    """使用预先计算的差值矩阵进行筛选（与 find_sequences 完全等价的结果）。

    - diff_arr: 形状为 N x C 的非负整型/浮点数组，表示 |length - target|；
    - tolerance: 标量或逐列容差向量；
    - 返回类型与顺序与 find_sequences 一致，以保证后续行为不变。
    """
    if isinstance(tolerance, (list, tuple, np.ndarray)):
        tol_vec = np.asarray(tolerance)
    else:
        # 容差标量时，按列广播
        tol_vec = tolerance
    mask = np.all(diff_arr <= tol_vec, axis=1)
    if isinstance(seqs_df, pd.DataFrame):
        return seqs_df[mask]
    else:
        return np.asarray(seqs_df)[mask]


def hamming_distance(s1, s2):
    """汉明距离计算（结果与原实现完全一致）。"""
    if len(s1) != len(s2):
        raise ValueError("Two strings have different lengths, cannot calculate Hamming distance")
    try:
        # 使用 NumPy 的 unpackbits 进行字节级 popcount 统计，加速大规模计算
        a = np.frombuffer(s1.encode('ascii'), dtype=np.uint8)
        b = np.frombuffer(s2.encode('ascii'), dtype=np.uint8)
        xorv = np.bitwise_xor(a, b)
        return int(np.unpackbits(xorv).sum())
    except Exception:
        # 回退到逐字符计算（与原实现一致）
        distance = 0
        for i in range(len(s1)):
            distance += bin(ord(s1[i]) ^ ord(s2[i])).count('1')
        return distance


# 预计算 0..255 的 popcount 查表，用于批量汉明距离
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distance_bulk(strings, query):
    """批量计算汉明距离（结果与逐个 hamming_distance 一致，顺序不变）。

    - strings: 序列列表，必须与 query 等长；
    - 返回 Python int 列表，顺序与输入一致；
    - 任意异常回退到逐元素计算，保证与原逻辑一致。
    """
    if not strings:
        return []
    qlen = len(query)
    for s in strings:
        if len(s) != qlen:
            raise ValueError("Two strings have different lengths, cannot calculate Hamming distance")
    try:
        # 构造二维字节数组：N x L
        # 通过拼接字节并 reshape，避免逐行堆叠开销
        joined = b"".join(s.encode('ascii') for s in strings)
        arr = np.frombuffer(joined, dtype=np.uint8).reshape(len(strings), qlen)
        q = np.frombuffer(query.encode('ascii'), dtype=np.uint8)
        xorv = np.bitwise_xor(arr, q)
        dists = _POPCOUNT_TABLE[xorv].sum(axis=1).astype(np.int32)
        return [int(x) for x in dists]
    except Exception:
        return [hamming_distance(s, query) for s in strings]


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
