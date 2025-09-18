import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import os
import atexit

# 全局线程/进程池缓存，避免频繁创建与销毁（减少进程创建时间）
_GLOBAL_THREAD_POOLS = {}
_GLOBAL_PROCESS_POOLS = {}


def _get_shared_thread_pool(max_workers):
    key = int(max_workers or (os.cpu_count() or 1))
    pool = _GLOBAL_THREAD_POOLS.get(key)
    if pool is None:
        pool = ThreadPoolExecutor(max_workers=key)
        _GLOBAL_THREAD_POOLS[key] = pool
    return pool


def _get_shared_process_pool(max_workers):
    key = int(max_workers or (os.cpu_count() or 1))
    pool = _GLOBAL_PROCESS_POOLS.get(key)
    if pool is None:
        # 不显式指定 spawn，默认在 Linux 为 fork，创建更快
        pool = ProcessPoolExecutor(max_workers=key)
        _GLOBAL_PROCESS_POOLS[key] = pool
    return pool


def _shutdown_global_pools():
    for pool in list(_GLOBAL_THREAD_POOLS.values()):
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _GLOBAL_THREAD_POOLS.clear()

    for pool in list(_GLOBAL_PROCESS_POOLS.values()):
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _GLOBAL_PROCESS_POOLS.clear()


atexit.register(_shutdown_global_pools)

class DynamicExecutor:
    def __init__(self, process_threshold=20, max_workers=None):
        """
        Initialize the dynamic executor.
        :param process_threshold: Threshold for the number of tasks to use ProcessPoolExecutor.
        :param max_workers: Maximum number of worker threads/processes.
        """
        self.process_threshold = process_threshold
        self.max_workers = max_workers or os.cpu_count()
        self._process_pool = None
        self._thread_pool = None
    
    def get_executor(self, task_count):
        """
        Return the appropriate executor based on the number of tasks.
        """
        if task_count >= self.process_threshold:
            if not self._process_pool:
                # 复用全局进程池，避免重复创建
                self._process_pool = _get_shared_process_pool(self.max_workers)
            return self._process_pool
        else:
            if not self._thread_pool:
                # 复用全局线程池
                self._thread_pool = _get_shared_thread_pool(self.max_workers)
            return self._thread_pool
    
    def shutdown(self):
        """Shut down all executors"""
        # 使用全局池时，不在此处关闭，避免重复创建销毁成本
        self._process_pool = None
        self._thread_pool = None

@contextmanager
def dynamic_executor_context(process_threshold=20, max_workers=None):
    """Context manager for automatic management of executor lifetime."""
    executor = DynamicExecutor(process_threshold, max_workers)
    try:
        yield executor
    finally:
        executor.shutdown()
