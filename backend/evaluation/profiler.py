# Background thread that samples CPU and memory usage during evaluation.

import time
import psutil
import threading
from typing import List
from .models import SystemProfile
from .logger import get_evaluation_logger

logger = get_evaluation_logger("profiler")

class SystemProfiler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        
        self.cpu_samples: List[float] = []
        self.mem_samples: List[float] = []
        
        self.initial_disk_read = 0
        self.initial_disk_write = 0
        self.final_disk_read = 0
        self.final_disk_write = 0

    def start(self):
        """Starts background system profiling."""
        self.running = True
        
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.initial_disk_read = disk_io.read_bytes
            self.initial_disk_write = disk_io.write_bytes
            
        self.thread = threading.Thread(target=self._profile_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("System profiling started.")

    def _profile_loop(self):
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.mem_samples.append(psutil.virtual_memory().used / (1024 * 1024)) # MB
            time.sleep(self.interval)

    def stop(self) -> SystemProfile:
        """Stops profiling and returns the summary."""
        self.running = False
        if self.thread:
            self.thread.join()
            
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.final_disk_read = disk_io.read_bytes
            self.final_disk_write = disk_io.write_bytes

        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0.0
        
        avg_mem = sum(self.mem_samples) / len(self.mem_samples) if self.mem_samples else 0.0
        peak_mem = max(self.mem_samples) if self.mem_samples else 0.0

        profile = SystemProfile(
            avg_cpu_usage=round(avg_cpu, 2),
            peak_cpu_usage=round(peak_cpu, 2),
            avg_memory_usage=round(avg_mem, 2),
            peak_memory_usage=round(peak_mem, 2),
            disk_read_bytes=self.final_disk_read - self.initial_disk_read,
            disk_write_bytes=self.final_disk_write - self.initial_disk_write
        )
        
        logger.info("System profiling stopped.")
        return profile
