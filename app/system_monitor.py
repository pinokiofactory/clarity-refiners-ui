import platform
import subprocess
import os
from typing import Optional, Dict, Tuple, Union
import psutil
import torch

NumericValue = Union[int, float]
MetricsDict = Dict[str, NumericValue]

class SystemMonitor:
    @staticmethod
    def get_nvidia_gpu_info() -> Tuple[str, MetricsDict]:
        """Get NVIDIA GPU information and metrics."""
        metrics = {}
        gpu_name = f"NVIDIA {torch.cuda.get_device_name(0)}"
        
        try:
            result = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,memory.reserved,temperature.gpu,utilization.gpu',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8', timeout=1.0)
            
            memory_used, memory_total, memory_reserved, temp, util = map(float, result.strip().split(','))
            metrics = {
                'memory_used': memory_used / 1024,  # Convert to GB
                'memory_total': memory_total / 1024,
                'memory_shared': memory_reserved / 1024,
                'temperature': temp,
                'utilization': util
            }
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback to basic CUDA info
            metrics = {
                'memory_used': torch.cuda.memory_allocated(0) / 1024**3,
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
            
        return gpu_name, metrics

    @staticmethod
    def get_mac_gpu_info() -> Tuple[str, MetricsDict]:
        """Get Apple Silicon GPU information."""
        metrics = {}
        try:
            # Get GPU memory info using powermetrics
            result = subprocess.check_output(
                ['powermetrics', '-n', '1', '--samplers', 'gpu_power'],
                encoding='utf-8'
            )
            
            if 'GPU Active Residency' in result:
                for line in result.split('\n'):
                    if 'GPU Active Residency' in line:
                        util = float(line.split(':')[1].strip().replace('%', ''))
                        metrics['utilization'] = util
            
            # Get basic system info for memory approximation
            memory_cmd = subprocess.check_output(['ps', '-caxm', '-orss,comm'], encoding='utf-8')
            metrics['memory_total'] = 8.0  # Typical unified memory allocation
            metrics['memory_used'] = sum(int(line.split()[0]) for line in memory_cmd.split('\n')[1:] 
                                       if line and 'GPU' in line) / 1024 / 1024
            
        except (subprocess.SubprocessError, FileNotFoundError):
            metrics = {
                'memory_total': 8.0,
                'memory_used': 0.0,
                'utilization': 0.0
            }
        
        return "Apple Silicon GPU", metrics

    @staticmethod
    def get_amd_gpu_info() -> Tuple[str, MetricsDict]:
        """Get AMD GPU information."""
        metrics = {}
        try:
            # Try rocm-smi first
            try:
                result = subprocess.check_output(['rocm-smi'], encoding='utf-8', timeout=1.0)
                
                for line in result.split('\n'):
                    if 'GPU Memory' in line:
                        parts = line.split()
                        used_idx = parts.index('Used') + 1
                        total_idx = parts.index('Total') + 1
                        metrics['memory_used'] = float(parts[used_idx]) / 1024
                        metrics['memory_total'] = float(parts[total_idx]) / 1024
                    elif 'Temperature' in line:
                        temp = float(line.split(':')[1].strip().replace('c', ''))
                        metrics['temperature'] = temp
                    elif 'GPU Utilization' in line:
                        util = float(line.split(':')[1].strip().replace('%', ''))
                        metrics['utilization'] = util
                        
            except (subprocess.SubprocessError, FileNotFoundError):
                # Try sysfs as fallback on Linux
                if platform.system() == "Linux":
                    base_path = "/sys/class/drm/card0/device"
                    try:
                        with open(f"{base_path}/hwmon/hwmon0/temp1_input") as f:
                            metrics['temperature'] = float(f.read().strip()) / 1000
                        
                        with open(f"{base_path}/mem_info_vram_total") as f:
                            metrics['memory_total'] = float(f.read().strip()) / 1024**3
                        with open(f"{base_path}/mem_info_vram_used") as f:
                            metrics['memory_used'] = float(f.read().strip()) / 1024**3
                            
                        with open(f"{base_path}/gpu_busy_percent") as f:
                            metrics['utilization'] = float(f.read().strip())
                    except (FileNotFoundError, PermissionError):
                        pass
        
        except Exception:
            metrics = {
                'memory_total': 0.0,
                'memory_used': 0.0,
                'utilization': 0.0
            }
        
        return "AMD GPU", metrics

    @staticmethod
    def is_amd_gpu() -> bool:
        """Check if system has an AMD GPU."""
        try:
            return (
                subprocess.call(['rocm-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
                or os.path.exists('/sys/class/drm/card0/device/vendor')
                and '0x1002' in open('/sys/class/drm/card0/device/vendor').read()
            )
        except:
            return False

    @classmethod
    def get_system_info(cls) -> str:
        """Get detailed system status with support for different GPU types."""
        try:
            # Determine GPU type and get metrics
            if torch.cuda.is_available():
                gpu_name, metrics = cls.get_nvidia_gpu_info()
            elif platform.system() == "Darwin" and platform.processor() == "arm":
                gpu_name, metrics = cls.get_mac_gpu_info()
            elif platform.system() == "Linux" and cls.is_amd_gpu():
                gpu_name, metrics = cls.get_amd_gpu_info()
            else:
                gpu_name, metrics = None, {}

            # Format GPU info based on available metrics
            if gpu_name:
                gpu_info = [f"🎮 GPU: {gpu_name}"]
                
                if 'memory_used' in metrics and 'memory_total' in metrics:
                    if platform.system() == "Darwin":
                        gpu_info.append(
                            f"📊 Unified Memory: {metrics['memory_used']:.1f}GB / {metrics['memory_total']:.1f}GB"
                        )
                    else:
                        gpu_info.append(
                            f"📊 GPU Memory: {metrics['memory_used']:.1f}GB / {metrics['memory_total']:.1f}GB"
                        )
                
                # Add shared memory for NVIDIA GPUs
                if 'memory_shared' in metrics:
                    gpu_info.append(f"💫 Shared Memory: {metrics['memory_shared']:.1f}GB")
                
                if 'temperature' in metrics:
                    gpu_info.append(f"🌡️ GPU Temp: {metrics['temperature']:.0f}°C")
                
                if 'utilization' in metrics:
                    gpu_info.append(f"⚡ GPU Load: {metrics['utilization']:.0f}%")
                    
                gpu_section = "\n".join(gpu_info) + "\n"
            else:
                gpu_section = "🎮 GPU: No dedicated GPU detected\n"
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            cpu_info = f"💻 CPU: {cpu_count} cores ({cpu_threads} threads)\n"
            cpu_usage = f"⚡ CPU Usage: {psutil.cpu_percent()}%\n"
            
            # Get RAM info
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            ram_info = f"🎯 RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram.percent}%)"
            
            return f"{gpu_section}{cpu_info}{cpu_usage}{ram_info}"
            
        except Exception as e:
            return f"Error collecting system info: {str(e)}"
