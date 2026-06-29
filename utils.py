import functools
import time  

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal total
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        total += duration
        print(f"Execution time for {func.__name__}: {duration:.2f}, Total: {total:.2f}")
        return result
    total = 0
    return wrapper