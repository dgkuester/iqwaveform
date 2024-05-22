import array_api_compat
from array_api_strict._typing import Array


def array_namespace(a):
    try:
        return array_api_compat.array_namespace(a)
    except TypeError:
        pass

    try:
        import mlx.core as mx
        if isinstance(a, mx.array):
            return mx
        else:
            raise TypeError
    except (ImportError, TypeError):
        pass

    raise TypeError('unrecognized object type')

