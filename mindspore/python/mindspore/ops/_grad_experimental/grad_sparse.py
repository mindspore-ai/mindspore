"""Define the grad rules of math related operations."""

from .. import functional as F
from .._grad.grad_base import bprop_getters
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations.sparse_ops import SparseTensorDenseAdd


@bprop_getters.register(SparseTensorDenseAdd)
def get_bprop_sparse_tensor_dense_add(self):
    """Grad definition for `SparseTensorDenseAdd` operation."""
    def bprop(x1_indices, x1_values, x1_shape, x2, out, dout):
        return (zeros_like(x1_indices), F.gather_nd(dout, x1_indices), zeros_like(x1_shape), dout,)
    return bprop
