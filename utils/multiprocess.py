import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import os

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
                self._process_pool = ProcessPoolExecutor(
                    max_workers=self.max_workers,
                )
            return self._process_pool
        else:
            if not self._thread_pool:
                self._thread_pool = ThreadPoolExecutor(
                    max_workers=self.max_workers
                )
            return self._thread_pool
    
    def shutdown(self):
        """Shut down all executors"""
        if self._process_pool:
            self._process_pool.shutdown()
            self._process_pool = None
        if self._thread_pool:
            self._thread_pool.shutdown()
            self._thread_pool = None

@contextmanager
def dynamic_executor_context(process_threshold=20, max_workers=None):
    """Context manager for automatic management of executor lifetime."""
    executor = DynamicExecutor(process_threshold, max_workers)
    try:
        yield executor
    finally:
        executor.shutdown()