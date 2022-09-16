# Copyright 2022 Huawei Technologies Co., Ltd
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

"""sparse_ops vmap impl."""
from __future__ import absolute_import

from mindspore.ops._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _raise_value_error
from mindspore.ops.primitive import Primitive
from mindspore.ops.operations.sparse_ops import DenseToCSRSparseMatrix, CSRSparseMatrixToSparseTensor, \
    CSRSparseMatrixToDense


@vmap_rules_getters.register(CSRSparseMatrixToSparseTensor)
def get_csr_sparse_matrix_to_sparse_tensor_vmap_rule(prim, axis_size):
    """VmapRule for `CSRSparseMatrixToSparseTensor` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(shape_bdim, x_batch_pointers_bdim, x_row_pointers_bdim, x_col_indices_bdim, x_values_bdim):
        is_all_none, result = vmap_general_preprocess(prim, shape_bdim, x_batch_pointers_bdim, x_row_pointers_bdim,
                                                      x_col_indices_bdim, x_values_bdim)
        if not is_all_none:
            _, shape_dim = shape_bdim
            _, x_batch_pointers_dim = x_batch_pointers_bdim
            _, x_row_pointers_dim = x_row_pointers_bdim
            _, x_col_indices_dim = x_col_indices_bdim
            _, x_values_dim = x_values_bdim
            _raise_value_error("For operator in CSRSparseMatrixToSparseTensor, all axes for inputs should be None, but"
                               " got shape_dim: {}, x_batch_pointesr_dim: {}, x_row_pointers_dim: {},"
                               " x_col_indices_dim: {}, and x_values_dim: {}.".format(shape_dim, x_batch_pointers_dim,
                                                                                      x_row_pointers_dim,
                                                                                      x_col_indices_dim, x_values_dim))
        return result

    return vmap_rule


@vmap_rules_getters.register(DenseToCSRSparseMatrix)
def get_dense_to_csr_sparse_matrix_vmap_rule(prim, axis_size):
    """VmapRule for `DenseToCSRSparseMatrix` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(dense_input_bdim, indices_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dense_input_bdim, indices_bdim)
        if not is_all_none:
            _, dense_input_dim = dense_input_bdim
            _, indices_dim = indices_bdim
            _raise_value_error("For operator in DenseToCSRSparseMatrix, all axes for inputs should be None, but"
                               " got dense_input_dim: {}, indices_dim: {}.".format(dense_input_dim, indices_dim))
        return result

    return vmap_rule


@vmap_rules_getters.register(CSRSparseMatrixToDense)
def get_csr_sparse_matrix_to_dense_vmap_rule(prim, axis_size):
    """VmapRule for `CSRSparseMatrixToDense` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(dense_shape_bdim, batch_ptr_bdim, row_ptr_bdim, col_indices_bdim, values_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dense_shape_bdim, batch_ptr_bdim, row_ptr_bdim,
                                                      col_indices_bdim, values_bdim)
        if not is_all_none:
            _, dense_shape_dim = dense_shape_bdim
            _, batch_ptr_dim = batch_ptr_bdim
            _, row_ptr_dim = row_ptr_bdim
            _, col_indices_dim = col_indices_bdim
            _, values_dim = values_bdim
            _raise_value_error("For operator CSRSparseMatrixToDense, all axes for inputs should be None, but got "
                               "dense_shape_dim: {} batch_ptr_dim: {} row_ptr_dim: {} col_indices_dim: {} and "
                               "values_dim: {}.".format(dense_shape_dim, batch_ptr_dim, row_ptr_dim, col_indices_dim,
                                                        values_dim))
        return result

    return vmap_rule
