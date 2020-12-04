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
"""cache_ops"""
from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register, PrimitiveWithCheck
from .. import signature as sig


class UpdateCache(PrimitiveWithCheck):
    """
    Update the value fo input_x, similar to ScatterNdUpdate.
    The diffirent is that UpdateCache will not update when indices < 0 or indices >= max_num.

    Inputs:
        - **input_x** (Parameter) - Parameter which is going to be updated.
        - **indices** (Tensor) - Update indices of input_x.
        - **updates** (Tensor) - The update values.

    Outputs:
        - **out** (Tensor) - Returns a [1] Tensor, which is not usefull.
    """
    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE,
                     dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T),
        sig.make_sig('max_num', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self):
        """init UpdateCache"""

        self.init_prim_io_names(inputs=['input_x', 'indices', 'update', 'max_num'],
                                outputs=['out'])

    def check_shape(self, input_x_shape, indices_shape, update_shape, max_num_shape):
        return [1]

    def check_dtype(self, input_x_dtype, indices_dtype, update_dtype, max_num_dtype):
        validator.check_tensor_dtype_valid(
            "indices", indices_dtype, mstype.int_type, self.name)
        return input_x_dtype


class SubAndFilter(PrimitiveWithCheck):
    """
    Dynamic kernel, sub an offset and
    return the elements which in range [0, max_num).

    Inputs:
        - **input_x** (Tensor) - Input tensor.
        - **max_num** (Int) - The max value of element that after sub `offset`.
        - **offset** (int) - Specifies the offset value of this `input_x`.

    Outputs:
        tuple(Tensor), tuple of 2 tensors, filter_res and filter_idx.
        - **filter_res** (Tensor) - The result that `input_x` minus `offset`,
          and return which in the range [0, max_num).
        - **filter_idx** (Tensor) - A tensor containing indices of elements in the input
          coressponding to the output tensor.

    Supported Platforms:
        `CPU`

    Examples:
        >>> x = Tensor(np.array([1, 3, 5, 8, 9, 16]), mindspore.int32)
        >>> max_num = 10
        >>> offset = 5
        >>> output = ops.SubAndFilter()(x, max_num, offset)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [0, 3, 4]),
         Tensor(shape=[3], dtype=Int32, value= [2, 3, 4]))
    """
    @prim_attr_register
    def __init__(self):
        """init SubAndFilter"""

        self.init_prim_io_names(inputs=['input_x', 'max_num', 'offset'],
                                outputs=['sub_res', 'sub_idx'])

    def check_shape(self, input_x_shape, max_num_shape, offset_shape):
        return (-1, -1)

    def check_dtype(self, input_x_dtype, max_num_dtype, offset_dtype):
        validator.check_tensor_dtype_valid(
            "input_x", input_x_dtype, mstype.int_type, self.name)
        return input_x_dtype


class SearchCacheIdx(PrimitiveWithInfer):
    """
    Search the keys of a hashmap, and return the values.

    Inputs:
        - **hashmap** (Parameter) - The dim of hashmap is (n, 4), which cols represent the `key, value, step, tag`.
        `key, value`: Map the indices of big table and cache table.
        `step`: The resent step, when searching the key, it will be updated at the same time.
        `step` can make sure the indices which are using in the last step will not be deleted in hashmap.
        `tag`: We use linear probing(`h(k, i) = (h(k) + i) % m`) to solve hash conflicts.
         tag is the count of linear probing times of the key. If `tag == 0`, means that the entry is empty.
        The Hash Function is:
         `((0.6180339 * key) - floor(0.618033 * key)) * hashmap_length`, in order to avoid data clustering.
        - **indices** (Tensor) - The indices which are keys of hashmap.
        - **step** (int) - The current step when searching.
        - **emb_max_num** (int) - Max length of big table.
         To avoid searching when `indices >= emb_max_num`, and make value = `cache_max_num`.
        - **cache_max_num** (int) - Max length of cache table.

    Outputs:
        - **cache_idx** (Tensor) - Result of searched value, if search missed, value = -1.
        - **miss_idx** (Tensor) - The index of Tensor indices which search missed.
         If search success, miss_idx[i] = -1.
        - **miss_emb_idx** (Tensor) - The value of Tensor indices which search missed.
         If search success, miss_emb_idx[i] = -1.
    Examples:
        >>> hashmap = Parameter(Tensor(np.array([[0, 0, 0, 0],
                                                [10, 5, -5, 1],
                                                [2, 1, -5, 1],
                                                [15, 7, -5, 2],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [3, 3, -5, 1],
                                                [21, 9, -5, 1]], np.int32)), name="hashmap")
        >>> indices = Tensor(np.array([10, 2, 25, 5, 3], np.int32))
        >>> step = 0, emb_max_num = 25, cache_max_num = 10
        >>> ops = ops.SearchCacheIdx()
        >>> cache_idx, miss_idx, miss_emb_idx = ops(hashmap, indices, step, emb_max_num, cache_max_num)
        cache_idx : [5, 1, 10, -1, 3]
        miss_idx : [-1, -1, -1, 3, -1]
        miss_emb_idx : [-1, -1, -1, 5, -1]
        hashmap after search : [[0, 0, 0, 0],
                                [10, 5, 0, 1],
                                [2, 1, 0, 1],
                                [15, 7, -5, 2],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [3, 3, 0, 1],
                                [21, 9, -5, 1]]
    """
    __mindspore_signature__ = (
        sig.make_sig('hashmap', sig.sig_rw.RW_WRITE,
                     dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T),
        sig.make_sig('step', dtype=sig.sig_dtype.T),
        sig.make_sig('emb_max_num', dtype=sig.sig_dtype.T),
        sig.make_sig('cache_max_num', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """init SearchCacheIdx"""

        self.init_prim_io_names(inputs=['hashmap', 'indices', 'step', 'emb_max_num', 'cache_max_num'],
                                outputs=['cache_idx', 'miss_idx', 'miss_emb_idx'])

    def infer_shape(self, hashmap_shape, indices_shape, step_shape, emb_max_num_shape, cache_max_num_shape):

        if len(hashmap_shape) != 2:
            raise ValueError("The dimension of 'hashmap' in SearchCacheIdx must be 2, "
                             "but got %d." % len(hashmap_shape))
        out_shape = (indices_shape, indices_shape, indices_shape)
        return out_shape

    def infer_dtype(self, hashmap_dtype, indices_dtype, step_dtype, emb_max_num_dtype, cache_max_num_dtype):
        args = {"hashmap": hashmap_dtype, "indices": indices_dtype}
        validator.check_tensors_dtypes_same_and_valid(
            args, mstype.int_type, self.name)
        out_dtype = (hashmap_dtype, hashmap_dtype, hashmap_dtype)
        return out_dtype


class MapUniform(PrimitiveWithCheck):
    """
    Map a tensor by using fomula : value = key % `group_num` * `per_group_size` + key // `group_num`.

    Inputs:
        - **input** (Tensor) - Input Tensor.
        - **per_group_size** (int) - The size of each group.
        - **group_num** (int) - The number of group.

    Outputs:
        Tensor, has the same dtype and shape as the `input`.

    Supported Platforms:
        `CPU`

    Examples:
        >>> input_x = Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        >>> per_group_size = 4
        >>> group_num = 2
        >>> map_uniform = ops.MapUniform()
        >>> output = map_uniform(input_x, per_group_size, group_num)
        >>> print(output)
        [0, 4, 1, 5, 2, 6, 3, 7]
    """

    @prim_attr_register
    def __init__(self):
        """init MapUniform"""
        self.init_prim_io_names(inputs=['input', 'per_group_size', 'group_num'],
                                outputs=['output'])

    def check_dtype(self, input_dtype, per_group_size_dtype, group_num_dtype):
        validator.check_tensor_dtype_valid(
            "input", input_dtype, mstype.int_type, self.name)
        validator.check_value_type(
            'per_group_size', per_group_size_dtype, [mstype.Int], self.name)
        validator.check_value_type(
            'group_num', group_num_dtype, [mstype.Int], self.name)


class CacheSwapHashmap(PrimitiveWithInfer):
    """
    Delete a hashmap entry,and insert a new key to hashmap, return the key and value of delete entry.

    Inputs:
        - **hashmap** (Parameter) - Same to operation SearchCacheIdx.
        - **miss_emb_idx** (Tensor) - The keys which are going to insert, -1 is skipped. It is the result
        - **step** (int) - The current step.

    Outputs:
        - **swap_cache_idx** (Tensor) - Deleted value of entry, -1 is skipped.
        - **old_emb_idx** (Tensor) - Deleted key of entry, -1 is skipped.
    """
    __mindspore_signature__ = (
        sig.make_sig('hashmap', sig.sig_rw.RW_WRITE,
                     dtype=sig.sig_dtype.T),
        sig.make_sig('miss_emb_idx', dtype=sig.sig_dtype.T),
        sig.make_sig('step', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """init CacheSwapHashmap"""

        self.init_prim_io_names(inputs=['hashmap', 'miss_emb_idx', 'step'],
                                outputs=['swap_cache_idx', 'old_emb_idx'])

    def infer_shape(self, hashmap_shape, miss_emb_idx_shape, step_shape):
        if len(hashmap_shape) != 2:
            raise ValueError("The dimension of 'hashmap' in CacheSwapHashmap must be 2, "
                             "but got %d." % len(hashmap_shape))

        out_shape = (miss_emb_idx_shape, miss_emb_idx_shape)
        return out_shape

    def infer_dtype(self, hashmap_dtype, miss_emb_idx_dtype, step_dtype):
        validator.check_tensor_dtype_valid(
            "miss_emb_idx", miss_emb_idx_dtype, mstype.int_type, self.name)
        out_dtype = (miss_emb_idx_dtype, miss_emb_idx_dtype)
        return out_dtype


class CacheSwapTable(PrimitiveWithCheck):
    """
    Delete a hashmap entry,and insert a new key to hashmap, return the key and value of delete entry.

    Inputs:
        - **cache_table** (Parameter) - The cache table which is on device.
        - **swap_cache_idx** (Tensor) - The index of table which need to swap. -1 is skipped.
        - **miss_value** (int) - The values which arg going to swap into cache table.

    Outputs:
        - **old_value** (Tensor) - The values which are swapped out.
    """
    __mindspore_signature__ = (
        sig.make_sig('cache_table', sig.sig_rw.RW_WRITE,
                     dtype=sig.sig_dtype.T),
        sig.make_sig('swap_cache_idx', dtype=sig.sig_dtype.T1),
        sig.make_sig('miss_value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """init CacheSwapTable"""

        self.init_prim_io_names(inputs=['cache_table', 'swap_cache_idx', 'miss_value'],
                                outputs=['old_value'])

    def check_shape(self, cache_table_shape, swap_cache_idx_shape, miss_value_shape):
        if len(cache_table_shape) != 2:
            raise ValueError(
                "cache table shape must be 2, but got %d" % len(cache_table_shape))

        return miss_value_shape

    def check_dtype(self, cache_table_dtype, swap_cache_idx_dtype, miss_value_dtype):
        validator.check_tensor_dtype_valid(
            "swap_cache_idx", swap_cache_idx_dtype, mstype.int_type, self.name)
        return miss_value_dtype


class MapCacheIdx(PrimitiveWithCheck):
    """
    MapCacheIdx merge SearchCacheIdx, CacheSwapHashmap, UpdateCache together.
    When input an indices tensor, it will output the cache indices which search in hashmap.
    """
    __mindspore_signature__ = (
        sig.make_sig('hashmap', sig.sig_rw.RW_WRITE,
                     dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T),
        sig.make_sig('step', dtype=sig.sig_dtype.T),
        sig.make_sig('emb_max_num', dtype=sig.sig_dtype.T),
        sig.make_sig('cache_max_num', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """init MapCacheIdx"""

        self.init_prim_io_names(inputs=['hashmap', 'indices', 'step', 'emb_max_num', 'offset'],
                                outputs=['cache_idx', 'old_emb_idx', 'miss_emb_idx', 'swap_cache_idx'])

    def __check__(self, hashmap, indices, step, emb_max_num, offset):
        hashmap_shape = hashmap['shape']
        if len(hashmap_shape) != 2:
            raise ValueError("The dimension of 'hashmap' in SearchCacheIdx must be 2, "
                             "but got %d." % len(hashmap_shape))
        out_shape = (indices['shape'], -1, -1, -1)

        hashmap_dtype = hashmap['dtype']
        indices_dtype = indices['dtype']
        args = {"hashmap": hashmap_dtype, "indices": indices_dtype}
        validator.check_tensors_dtypes_same_and_valid(
            args, mstype.int_type, self.name)
        out_dtype = (hashmap_dtype, hashmap_dtype,
                     hashmap_dtype, hashmap_dtype)

        out = {'shape': out_shape,
               'dtype': out_dtype,
               'value': None}
        if 'max_shape' in indices:
            out['max_shape'] = (indices['max_shape'], indices['max_shape'],
                                indices['max_shape'], indices['max_shape'])
        else:
            out['max_shape'] = (indices['shape'], indices['shape'],
                                indices['shape'], indices['shape'])
        if 'min_shape' in indices:
            out['min_shape'] = (indices['min_shape'], 0, 0, 0)
        else:
            out['min_shape'] = (0, 0, 0, 0)
        return out
