from main import train
import torch
import subprocess
import gc
import sys
import psutil
from util.memory_gpu_usage import get_gpu_memory_usage

#import multiprocessing
#from multiprocessing import Pool
#from multiprocessing import Pool, TimeoutError
#import contextlib

#import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
import numpy as np
import time
import os

from option import lst_parameters_change

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
def daemon(index):
    print 'Starting:', multiprocessing.current_process().name
    #time.sleep(2)
    print('Index %d' % index)
    #train(index)
    print 'Exiting :', multiprocessing.current_process().name

def non_daemon(index):
    print 'Starting:', multiprocessing.current_process().name
    print('Index %d' % index)
    train(index)
    print 'Exiting :', multiprocessing.current_process().name

if __name__ == '__main__':
    n = multiprocessing.Process(name='non-daemon', target=non_daemon, args=(2,))
    n.daemon = False

    n.start()
    n.join()
    print('... Finished')
'''

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def work(num_proc):
    print("Creating %i (daemon) worker." % num_proc)
    try:
        train(index=num_proc)
    except:
        print('Exception in worker: %d',num_proc)
        return False
    return True

def script():
    max_pool_tasks = 3
    print("Creating 5 (non-daemon) workers and jobs in main process.")
    max_tasks = len(lst_parameters_change)
    if max_pool_tasks > max_tasks:
        max_pool_tasks = max_tasks

    for i in np.arange(0,10,max_pool_tasks):

        range_ini_task = i
        range_end_task = i + max_pool_tasks

        if range_end_task >= max_tasks:
            range_end_task = max_tasks

        pool = NoDaemonPool(max_pool_tasks)
        pool.map(work, range(range_ini_task,range_end_task))
        pool.close()
        pool.join()

        # Finish
        if range_end_task >= max_tasks:
            break

if __name__ == "__main__":
    script()