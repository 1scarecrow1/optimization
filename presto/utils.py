from functools import wraps, partial
import time  
import logging 
from collections.abc import Callable
import warnings

logger = logging.getLogger(__name__)

def resolve_func(f, methods: dict = None, name='', default=None):
    '''
    default must be a callable
    '''
    if f is None and default is not None:
        return default
    if isinstance(f, str):
        if methods and f in methods:
            return methods[f]
        if default is not None:
            warnings.warn(f"{name} method {f!r} not available, using "
                f"{getattr(default, '__name__', default)}", stacklevel=2)
            return default

        valid = ", ".join(methods or ())
        raise NotImplementedError(
            f"{name} method {f!r} is either not implemented, not valid, or not in {methods}. "
            f"Pass a function yourself or choose one of: {valid}"
        )

    elif isinstance(f, Callable):                    
        func = f.func if isinstance(f, partial) else f
        if not methods or func in methods.values() or default is None:
            return f                                 
    
    if default is not None:
        warnings.warn(f"{name} expected a method name or callable, got "
                      f"{type(f).__name__}; using "
                      f"{getattr(default, '__name__', default)}", stacklevel=2)
        return default
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

def timer2(func, log=print, enabled=True):
    @wraps(func)
    def wrapper(*args, **kwargs):
        total = 0
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        total += duration
        print(f"Execution time for {func.__name__}: {duration:.2f}, Total: {total:.2f}")
        return result
    return wrapper

def timer(func=None, *, log=logger.debug, enabled=True):
    '''
    Time every call to func. Stats accumulate on the wrapper itself:

        minimize.calls    calls since the last reset
        minimize.total    summed wall time
        minimize.last     duration of the most recent call
        minimize.reset()  zero the counters -- call between benchmark runs
        minimize.enabled  set False to silence and skip the timing entirely

    Pass log=logger.debug to route through logging instead of stdout.
    '''
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not wrapper.enabled:
                return f(*args, **kwargs)
            start = time.perf_counter()
            try:
                return f(*args, **kwargs)
            finally:                                  # times the failing call too
                d = time.perf_counter() - start
                wrapper.last = d
                wrapper.calls += 1
                wrapper.total += d
                log(f"{f.__name__}: {_fmt(d)} "
                    f"(call {wrapper.calls}, total {_fmt(wrapper.total)})")
        wrapper.calls = 0                             
        wrapper.total = 0.0
        wrapper.last = None
        wrapper.enabled = enabled
        wrapper.reset = lambda: wrapper.__dict__.update(calls=0, total=0.0, last=None)
        return wrapper
    return decorate(func) if func is not None else decorate

def _fmt(seconds):
    if seconds < 1e-3:
        return f"{seconds:.6f}s"
    if seconds < 1.0:
        return f"{seconds:.6f}s"
    return f"{seconds:.3f}s"   