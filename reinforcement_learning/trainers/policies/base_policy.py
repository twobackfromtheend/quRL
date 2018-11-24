from typing import Callable, Union


class BasePolicy:
    @classmethod
    def get_action(cls, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def log_if(log_func: Union[None, Callable], message: str):
        """
        Conditional logging that allows log_func to be None.
        Call with
            self.log_if(log_func, message)
        :param log_func:
        :param message:
        :return:
        """
        if log_func is not None:
            log_func(message)
