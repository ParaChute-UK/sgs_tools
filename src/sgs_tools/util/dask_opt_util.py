import dask
from functools import wraps

def dask_layered(fn_or_lbl=None):
    def decorator(fn, lbl=None):
      @wraps(fn)
      def wrapper(*args, **kwargs):
          layer_name = lbl or fn.__name__
          with dask.annotate(layer=layer_name):
              return fn(*args, **kwargs)
      return wrapper
    if callable(fn_or_lbl):  # used as @dask_layered
        return decorator(fn_or_lbl)
    else:  # used as @dask_layered("label")
        return lambda fn: decorator(fn, lbl=fn_or_lbl)