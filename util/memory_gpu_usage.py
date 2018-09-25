import torch
import subprocess
import os
import gc
import sys
import psutil

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())

def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

def get_gpu_memory_usage(id):
    """Get the current gpu usage.
    """
    # info = subprocess.check_output(['nvidia-smi', '-i', '0', '-q', '-d', 'MEMORY'])
    gpu_name = subprocess.check_output(
        ['nvidia-smi', '-i', str(id), '--query-gpu=gpu_name', '--format=csv,nounits,noheader']).split('\n')[0]
    use_memory = subprocess.check_output(
        ['nvidia-smi', '-i', str(id), '--query-gpu=utilization.memory', '--format=csv,nounits,noheader']).split('\n')[0]
    memory_total = subprocess.check_output(
        ['nvidia-smi', '-i', str(id), '--query-gpu=memory.total', '--format=csv,nounits,noheader']).split('\n')[0]
    memory_free = subprocess.check_output(
        ['nvidia-smi', '-i', str(id), '--query-gpu=memory.free', '--format=csv,nounits,noheader']).split('\n')[0]
    memory_used = subprocess.check_output(
        ['nvidia-smi', '-i', str(id), '--query-gpu=memory.used', '--format=csv,nounits,noheader']).split('\n')[0]
    return gpu_name, int(use_memory), int(memory_total), int(memory_free), int(memory_used)