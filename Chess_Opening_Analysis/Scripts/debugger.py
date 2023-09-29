# Script for debugging functions

import time

def execution_time(func):
    start = time.perf_counter()
    result = func()
    finish = time.perf_counter()
    execution_time = finish - start
    print(execution_time)
    return result, execution_time
