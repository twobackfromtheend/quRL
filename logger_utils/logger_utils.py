import time
from logging import Logger
from typing import Callable

get_time_func = time.process_time


def log_process(logger: Logger, process_name: str):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            logger.info(f'Started {process_name}.')
            start_time = get_time_func()

            return_value = func(*args, **kwargs)

            end_time = get_time_func()
            duration = end_time - start_time
            logger.info(f'Completed {process_name}. (Time taken: {duration:.3f}s)')
            return return_value

        return wrapper

    return decorator
