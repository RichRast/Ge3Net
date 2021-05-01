import time
import functools
from utils.dataUtil import square_normalize

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time    
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

# deprecate/ not used
def applySquareNormalize(geography):
    def innerSquareNormalize(func):
        """
        apply square normalizing the output if
        the parameter of geography is true
        """
        @functools.wraps(func)
        def wrapper_squareNormalize(*args, **kwargs):
            value = func(*args, **kwargs)
            if geography: return square_normalize(value)
            else :return value

        return wrapper_squareNormalize
    return innerSquareNormalize
