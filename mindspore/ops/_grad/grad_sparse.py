# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""bprop primitives"""
from .. import functional as F
from .. import operations as P
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import bprops, bprop_getters

# Unused parameters are placeholders.


@bprops.register("MakeSparseTensor")
def bprop_make_sparse_tensor(indices, values, dense_shape, out, dout):
    """Backpropagator for primitive `MakeSparseTensor`."""
    return zeros_like(indices), F.sparse_tensor_get_values(dout), ()


@bprops.register("SparseTensorGetIndices")
def bprop_sparse_tensor_get_indices(sparse_tensor, out, dout):
    """Backpropagator for primitive `SparseTensorGetIndices`."""
    return (zeros_like(sparse_tensor),)


@bprops.register("SparseTensorGetValues")
def bprop_sparse_tensor_get_values(sparse_tensor, out, dout):
    """Backpropagator for primitive `SparseTensorGetValues`."""
    return F.make_sparse_tensor(F.sparse_tensor_get_indices(sparse_tensor),
                                dout,
                                F.sparse_tensor_get_dense_shape(sparse_tensor))


@bprops.register("SparseTensorGetDenseShape")
def bprop_sparse_tensor_get_dense_shape(sparse_tensor, out, dout):
    """Backpropagator for primitive `SparseTensorGetDenseShape`."""
    return (zeros_like(sparse_tensor),)


@bprop_getters.register(P.SparseToDense)
def get_bprop_sparse_to_dense(self):
    """Generate bprop for SparseToDense"""

    def bprop(indices, values, dense_shape, out, dout):
        return zeros_like(indices), dout, zeros_like(dense_shape)

    return bprop
