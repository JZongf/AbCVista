from time import time

def time_warpper(func, func_name=None, fstream=None):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        if func_name is None:
            func_name = func.__name__
        if fstream is None:
            print(f"{func_name} costs {end_time - start_time} seconds.")
        else:
            fstream.write(f"{func_name}, {end_time - start_time}\n")
        return result
    return wrapper