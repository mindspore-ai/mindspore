"""
type assert function
"""

from inspect import signature
from functools import wraps


def typeassert(*ty_args, **ty_kwargs):
    """
    type assert function
    """
    def decorate(func):
        # If in optimized mode, disable type checking
        if not __debug__:
            return func

        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            f'Argument {name} must be {bound_types[name]}'
                            )
            return func(*args, **kwargs)
        return wrapper
    return decorate
