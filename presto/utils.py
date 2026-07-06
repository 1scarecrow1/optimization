from functools import wraps, partial
import time  
from collections.abc import Callable

def resolve_func(f, methods: dict = None, name=''):
    if isinstance(f, str):
        try:
            return methods[f]
        except KeyError:
            valid = ", ".join(methods)
            raise NotImplementedError(
                f"{name} method {f!r} is either not implemented or not in {methods}. "
                f"Define a function yourself or choose one of: {valid}"
            ) from None

    if isinstance(f, partial):
        return f.func 
    if isinstance(f, Callable):
        return f
    raise TypeError(
        f"{name} expected a method name string or callable, "
        f"got {type(f).__name__}"
    )

def merge_args(*dicts, check_dict=""):
    merged = {}
    for d in dicts:
        dup = merged.keys() & d.keys()
        if dup:
            conflicts = ", ".join(f"{k!r}: {merged[k]!r} vs {d[k]!r}" for k in sorted(dup))
            where = f" in {check_dict}" if check_dict else ""
            raise ValueError(f"conflicting keyword args {where} ({conflicts})")
        merged.update(d)
    return merged

def timer(func):
    @wraps(func)
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