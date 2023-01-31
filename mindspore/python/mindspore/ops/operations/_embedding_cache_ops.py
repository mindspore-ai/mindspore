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

from __future__ import absolute_import
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import prim_attr_register, PrimitiveWithCheck
from mindspore.ops import signature as sig


class UpdateCache(PrimitiveWithCheck):
    """
    Update the value fo input_x, similar to ScatterNdUpdate.
    The difference is that UpdateCache will not update when indices < 0 or indices >= max_num.

    Inputs:
        - **input_x** (Parameter) - Parameter which is going to be updated.
        - **indices** (Tensor) - Update indices of input_x.
        - **updates** (Tensor) - The update values.

    Outputs:
        - **out** (Tensor) - Returns a [1] Tensor, which is not useful.
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
        return ((-1,), (-1,))

    def check_dtype(self, input_x_dtype, max_num_dtype, offset_dtype):
        validator.check_tensor_dtype_valid(
            "input_x", input_x_dtype, mstype.int_type, self.name)
        return input_x_dtype


class MapUniform(PrimitiveWithCheck):
    """
    Map a tensor by using formula : value = key % `group_num` * `per_group_size` + key // `group_num`.

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

        return out


class DynamicAssign(PrimitiveWithCheck):
    """
    Assigns `Parameter` with a value, the `value` can have a dynamic shape.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Tensor) - The value to be assigned.

    Outputs:
        Tensor, has the same type as original `variable`.

    Supported Platforms:
        `CPU`
    """
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['output'])

    def check_dtype(self, variable, value):
        if variable != mstype.type_refkey:
            validator.check_tensor_dtype_valid(
                "variable", variable, mstype.number_type, self.name)
        validator.check_scalar_or_tensor_types_same(
            {"value": value}, mstype.number_type, self.name)


class PadAndShift(PrimitiveWithCheck):
    """
    Initialize a tensor with -1, and copy a slice from `input_x` to the padded Tensor.

    Note:
        If use python, PadAndShift is:
            output = [-1] * cum_sum_arr[-1]
            start = cum_sum_arr[shift_idx]
            end = cum_sum_arr[shift_idx + 1]
            output[start:end] = input_x[:(end-start)]

    Inputs:
        - **input_x** (Tensor) - The input Tensor, which will be copied
          to `output`.
        - **cum_sum_arr** (Tensor) - The last value of cum_sum_arr is
          the pad length of output tensor, `cum_sum_arr[shift_idx]` is
          the start to shift, and `cum_sum_arr[shift_idx+1]` is the end.
        - **shift_idx** (int) - The idx of `cum_sum_arr` .

    Outputs:
        - **output** (Tensor) - Tensor, has the same type as `input`.

    Raises:
        TypeError: `input_x` or `cum_sum_arr` is not Tensor.
        TypeError: `shift_idx` is not int.
        ValueError: Value of `shift_idx` is larger than or equal to the length of `cum_sum_arr` .

    Supported Platforms:
        `CPU`

    Examples:
        >>> input_x = Tensor(np.array([9, 13, -1, -1, -1, -1, -1, -1]), mstype.int32)
        >>> cum_sum_arr = Tensor(np.array([0, 3, 5]), mstype.int32)
        >>> shift_idx = 1
        >>> pad_and_shift = ops.PadAndShift()
        >>> output = pad_and_shift(input_x, cum_sum_arr, shift_idx)
        >>> print(output)
        [-1, -1, -1, 9, 13]
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(
            inputs=['input_x', 'cum_sum_arr', 'shift_idx'], outputs=['output'])

    def check_shape(self, input_x_shape, cum_sum_arr_shape, shift_idx_shape):
        return input_x_shape

    def check_dtype(self, input_x_dtype, cum_sum_arr_dtype, shift_idx_dtype):
        return input_x_dtype
