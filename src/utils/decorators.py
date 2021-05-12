import time
import functools
# from src.utils.dataUtil import square_normalize

EPS_CONST=1e-7
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

def guardAgainstDivideZero(operate):
    @functools.wraps(operate)
    def inner(x, y):
        if y<EPS_CONST:
            raise ZeroDivisionError("Check the denominator, it is close to zero")
        return operate(x,y)
    return inner

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
            valueNormalized=square_normalize(value)
            if geography: return valueNormalized
            else :return value

        return wrapper_squareNormalize
    return innerSquareNormalize
