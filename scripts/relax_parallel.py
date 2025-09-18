#!/usr/bin/env python
# 独立的 relax 并行执行脚本
import os
import sys
import json
import pickle
import argparse
import time
import logging
import subprocess
from typing import List, Dict, Optional, Tuple

script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, script_dir)

from openfold.np import protein
from openfold.config import model_config
import openfold.np.relax.relax as relax

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def _detect_available_gpus() -> List[str]:
    """检测可用 GPU 列表，返回可见的 GPU 索引字符串列表，如 ["0", "1"]."""
    # 优先使用 CUDA_VISIBLE_DEVICES；否则使用 torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        # 形如 "0,1,3" -> ["0","1","3"]
        return [x for x in visible.split(",") if x != ""]
    try:
        import torch  # 延迟导入，避免无 GPU 环境初始化开销
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)]


def _build_proc_env(base_env: Dict[str, str], *, gpu_id: Optional[str], cpu_threads: int, use_gpu: bool) -> Dict[str, str]:
    """构建子进程环境变量
    - 限制每个进程的 BLAS/OMP 线程数，避免过度抢占 CPU
    - 为 GPU 进程设置独占的 CUDA_VISIBLE_DEVICES
    """
    env = dict(base_env)
    # 限制线程数，提升多进程整体吞吐（强制覆盖，避免父进程遗留配置导致过度并行）
    cpu_threads = max(int(cpu_threads), 1)
    env["OMP_NUM_THREADS"] = str(cpu_threads)
    env["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    env["MKL_NUM_THREADS"] = str(cpu_threads)
    env["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    # OpenMM CPU 平台线程数
    env["OPENMM_CPU_THREADS"] = str(cpu_threads)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # 关闭 Python 缓冲，确保日志及时输出
    env["PYTHONUNBUFFERED"] = "1"

    if use_gpu:
        # 为该子进程绑定单块 GPU；在子进程内设备将被重映射为 cuda:0
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # 强制不使用 GPU
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _run_single_relax_task(*, pkl_path: str, output_directory: str, output_name: str,
                           config_preset: str, model_device: str, cif_output: bool,
                           config_cache: Dict[str, object], relaxer_cache: Dict[Tuple[str, str], relax.AmberRelaxation]) -> Tuple[bool, float, Optional[str]]:
    """执行一个 relax 任务（在当前进程内）。
    返回 (ok, elapsed, err_msg)。
    说明：缓存 config 与 AmberRelaxation，减少重复初始化开销。
    """
    # 取/建 config（缓存）
    if config_preset not in config_cache:
        config_cache[config_preset] = model_config(config_preset)
    cfg = config_cache[config_preset]

    # 取/建 relaxer（缓存 key: (device, preset)）
    device_key = "cpu" if model_device == "cpu" else "cuda"
    relax_key = (device_key, config_preset)
    if relax_key not in relaxer_cache:
        # 创建 amber relaxer（只在首次构建时开销较大）
        relaxer_cache[relax_key] = relax.AmberRelaxation(
            use_gpu=(model_device != "cpu"),
            **cfg.relax,
        )
    amber_relaxer = relaxer_cache[relax_key]

    # 加载未 relax 的蛋白质结构
    with open(pkl_path, 'rb') as f:
        unrelaxed_protein = pickle.load(f)

    t0 = time.perf_counter()
    try:
        struct_str, _, _ = amber_relaxer.process(
            prot=unrelaxed_protein,
            cif_output=cif_output,
        )
        elapsed = time.perf_counter() - t0

        # 保存 relaxed 结构
        suffix = "_relaxed.cif" if cif_output else "_relaxed.pdb"
        relaxed_output_path = os.path.join(output_directory, f"{output_name}{suffix}")
        with open(relaxed_output_path, 'w') as fp:
            fp.write(struct_str)

        # 更新计时信息
        try:
            timings_path = os.path.join(output_directory, "timings.json")
            if os.path.exists(timings_path):
                with open(timings_path, 'r') as f:
                    timings = json.load(f)
            else:
                timings = {}
            timings[f"relaxation_{output_name}"] = elapsed
            with open(timings_path, 'w') as f:
                json.dump(timings, f, indent=2)
        except Exception:
            pass

        logger.info(f"Relaxation completed: {output_name}, time={elapsed:.2f}s")
        return True, elapsed, None
    except Exception as e:
        logger.error(f"Relaxation failed for {output_name}: {e}")
        return False, time.perf_counter() - t0, str(e)


def parallel_relax_proteins(relax_tasks, model_device=None, cpus=4, max_workers=None):
    """
    并行执行多个 relax 任务
    
    Args:
        relax_tasks: 列表，包含所有需要 relax 的任务参数
        model_device: 模型设备；为 None 或 'auto' 时自动检测（有 GPU 则用 GPU，否则用 CPU）
        cpus: CPU 核心数
        max_workers: 最大并行进程数，默认使用 CPU 核心数
    """
    if not relax_tasks:
        return
    
    # 从第一个任务的 pkl 路径推断临时目录位置
    temp_dir = None
    if relax_tasks and 'pkl_path' in relax_tasks[0]:
        pkl_dir = os.path.dirname(relax_tasks[0]['pkl_path'])
        if os.path.basename(pkl_dir) == 'temp_relax_pkl':
            temp_dir = pkl_dir
    
    try:
        # 计算最大并行数与 GPU 拓扑
        total_wall_start = time.perf_counter()  # 统计整个 relax 阶段耗时（墙钟时间）
        if model_device is None or str(model_device).lower() in {"auto", "", "none"}:
            gpu_list = _detect_available_gpus()
            use_gpu = len(gpu_list) > 0
        else:
            use_gpu = ("cuda" in str(model_device).lower())
            gpu_list = _detect_available_gpus() if use_gpu else []

        if max_workers is None:
            if use_gpu:
                # GPU 模式：默认与可用 GPU 数量一致
                max_workers = max(len(gpu_list), 1)
            else:
                # CPU 模式：保守限制，避免过多上下文切换
                max_workers = max(min(int(cpus), 4), 1)

        # GPU 模式允许单卡多进程并行（共享同一 GPU），以满足单 GPU 并行需求
        # 注意：多进程共享 GPU 可能引发上下文切换与显存抢占，需通过 --relax_max_workers 控制并发度
        effective_workers = max_workers

        # 每进程 CPU 线程数（均分，至少为 1）
        cpu_threads_per_worker = max(int(max(cpus, 1) / max(effective_workers, 1)), 1)

        logger.info(
            f"Starting parallel relaxation: tasks={len(relax_tasks)}, workers={effective_workers}, "
            f"gpu={use_gpu} (single-GPU may time-slice), visible_gpus={','.join(gpu_list) if gpu_list else 'none'}, "
            f"cpu_threads/worker={cpu_threads_per_worker}"
        )

        # 统一采用 Popen + 有界并发调度（逐任务），避免在进程池里再启子进程的额外开销
        running: List[Dict] = []  # [{proc, name, start_time}]
        pending: List[Dict] = []

        for idx, task in enumerate(relax_tasks):
            # 参数
            pkl_path = task['pkl_path']
            output_name = task['output_name']
            output_directory = task['output_directory']
            config_preset = task['config_preset']
            # 设备由父进程统一调度，无需在任务中指定
            cif_output = task['cif_output']

            # 构建命令
            cmd = [
                sys.executable,
                __file__,
                "--unrelaxed_pkl", pkl_path,
                "--output_directory", output_directory,
                "--output_name", output_name,
                "--config_preset", config_preset,
                # 子进程内设备可见性被重映射为 0，这里统一传 cuda:0（或 cpu）
                "--model_device", ("cuda:0" if use_gpu else "cpu"),
            ]
            if cif_output:
                cmd.append("--cif_output")

            # 为该任务选择 GPU（如启用 GPU）
            gpu_id = None
            if use_gpu:
                if len(gpu_list) == 0:
                    gpu_id = None  # 兼容性处理
                else:
                    gpu_id = gpu_list[idx % len(gpu_list)]

            # 组装子进程环境
            env = _build_proc_env(
                os.environ,
                gpu_id=gpu_id,
                cpu_threads=cpu_threads_per_worker,
                use_gpu=use_gpu,
            )

            pending.append({
                'cmd': cmd,
                'env': env,
                'name': output_name,
            })

        completed = 0
        failed = 0

        def _launch_next():
            if not pending:
                return False
            item = pending.pop(0)
            proc = subprocess.Popen(
                item['cmd'],
                env=item['env'],
                stdout=None,
                stderr=None,
                text=True,
            )
            running.append({
                'proc': proc,
                'name': item['name'],
                'start': time.time(),
            })
            logger.info(f"Launched relax: {item['name']}")
            return True

        # 初始填满并发槽位
        while len(running) < effective_workers and _launch_next():
            pass

        # 轮询已启动的任务，维持并发
        while running:
            # 遍历拷贝，便于在循环内修改 running 列表
            for entry in list(running):
                proc: subprocess.Popen = entry['proc']
                ret = proc.poll()
                if ret is None:
                    continue  # 仍在运行

                running.remove(entry)
                if ret == 0:
                    completed += 1
                else:
                    failed += 1

                # 补充启动下一个任务
                if len(running) < effective_workers:
                    _launch_next()

            # 降低轮询频率
            time.sleep(0.2)

        total_wall = time.perf_counter() - total_wall_start
        logger.info(
            f"Parallel relaxation finished: {completed} completed, {failed} failed, total_wall_time={total_wall:.2f}s"
        )
    finally:
        # 清理临时目录及其内容
        if temp_dir and os.path.exists(temp_dir):
            try:
                # 删除目录中的所有 pkl 文件
                for pkl_file in os.listdir(temp_dir):
                    if pkl_file.endswith('.pkl'):
                        os.remove(os.path.join(temp_dir, pkl_file))
                
                # 现在删除空目录
                if not os.listdir(temp_dir):  # 确认目录为空
                    os.rmdir(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception:
                # 静默忽略错误，不影响主流程
                pass

def relax_single_protein(args):
    """执行单个蛋白质的 relax 过程"""
    
    # 复用批量执行的核心逻辑，避免重复代码
    ok, elapsed, err = _run_single_relax_task(
        pkl_path=args.unrelaxed_pkl,
        output_directory=args.output_directory,
        output_name=args.output_name,
        config_preset=args.config_preset,
        model_device=args.model_device,
        cif_output=args.cif_output,
        config_cache={},
        relaxer_cache={},
    )
    logger.info(f"Relaxation time: {elapsed:.2f}s")
    return 0 if ok else 1

def main():
    parser = argparse.ArgumentParser(description='Relax a single protein structure')
    
    parser.add_argument('--unrelaxed_pkl', type=str, required=True,
                        help='Path to pickled unrelaxed protein')
    parser.add_argument('--output_directory', type=str, required=True,
                        help='Directory to save relaxed structure')
    parser.add_argument('--output_name', type=str, required=True,
                        help='Name for the output file (without extension)')
    parser.add_argument('--config_preset', type=str, required=True,
                        help='Model config preset')
    parser.add_argument('--model_device', type=str, default='cuda:0',
                        help='Device for relaxation')
    parser.add_argument('--cif_output', action='store_true',
                        help='Output in CIF format instead of PDB')
    # 批量模式参数（仅在并行调度内部使用）
    parser.add_argument('--batch_file', type=str, default=None,
                        help='JSON file containing a list of relax tasks for batch worker')
    parser.add_argument('--result_file', type=str, default=None,
                        help='Path to JSON results written by batch worker')
    
    args = parser.parse_args()
    
    # 执行 relax
    exit_code = relax_single_protein(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
