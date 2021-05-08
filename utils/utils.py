from functools import wraps
from datetime import datetime

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'{func.__qualname__} starting')
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print('{} complete. elapsed time {}'.format(
            func.__name__, end-start))
        return result
    return wrapper