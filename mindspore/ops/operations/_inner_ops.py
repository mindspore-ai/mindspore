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

"""Inner operators."""

from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import  PrimitiveWithInfer, prim_attr_register


class ExtractImagePatches(PrimitiveWithInfer):
    """
    Extract patches from images.
    The input tensor must be a 4-D tensor and the data format is NHWC.

    Args:
        ksizes (Union[tuple[int], list[int]]): The size of sliding window, should be a tuple or list of int,
            and the format is [1, ksize_row, ksize_col, 1].
        strides (Union[tuple[int], list[int]]): Distance between the centers of the two consecutive patches,
            should be a tuple or list of int, and the format is [1, stride_row, stride_col, 1].
        rates (Union[tuple[int], list[int]]): In each extracted patch, the gap between the corresponding dim
            pixel positions, should be a tuple or list of int, and the format is [1, rate_row, rate_col, 1].
        padding (str): The type of padding algorithm, is a string whose value is "same" or "valid",
            not case sensitive. Default: "valid".

            - same: Means that the patch can take the part beyond the original image, and this part is filled with 0.

            - valid: Means that the patch area taken must be completely contained in the original image.

    Inputs:
        - **input_x** (Tensor) - A 4-D tensor whose shape is [in_batch, in_row, in_col, in_depth] and
          data type is number.

    Outputs:
        Tensor, a 4-D tensor whose data type is same as 'input_x',
        and the shape is [out_batch, out_row, out_col, out_depth], the out_batch is same as the in_batch.
    """

    @prim_attr_register
    def __init__(self, ksizes, strides, rates, padding="valid"):
        """init"""
        def _check_tuple_or_list(arg_name, arg_val, prim_name):
            validator.check_value_type(f"{arg_name}s", ksizes, [tuple, list], self.name)
            if len(arg_val) != 4 or arg_val[0] != 1 or arg_val[3] != 1:
                raise ValueError(f"For \'{prim_name}\' the format of {arg_name}s should be [1, {arg_name}_row, "
                                 f"{arg_name}_col, 1], but got {arg_val}.")
            if not isinstance(arg_val[1], int) or not isinstance(arg_val[2], int) or arg_val[1] < 1 or arg_val[2] < 1:
                raise ValueError(f"For '{prim_name}' the {arg_name}_row and {arg_name}_col in {arg_name}s should be an "
                                 f"positive integer number, but got {arg_name}_row is {arg_val[1]}, {arg_name}_col "
                                 f"is {arg_val[2]}")

        _check_tuple_or_list("ksize", ksizes, self.name)
        _check_tuple_or_list("stride", strides, self.name)
        _check_tuple_or_list("rate", rates, self.name)
        self.padding = validator.check_string('padding', padding.upper(), ['VALID', 'SAME'], self.name)
        self.add_prim_attr("padding", self.padding)

    def infer_shape(self, input_x):
        """infer shape"""
        in_batch, in_row, in_col, in_depth = input_x
        _, ksize_row, ksize_col, _ = self.ksizes
        _, stride_row, stride_col, _ = self.strides
        _, rate_row, rate_col, _ = self.rates
        if len(input_x) != 4:
            raise ValueError("The `input_x` should be a 4-D tensor, "
                             f"but got a {len(input_x)}-D tensor whose shape is {input_x}")

        out_batch = in_batch
        out_depth = ksize_row * ksize_col * in_depth

        if self.padding == "VALID":
            out_row = \
                (in_row - (ksize_row + (ksize_row - 1) * (rate_row - 1))) // stride_row + 1
            out_col = \
                (in_col - (ksize_col + (ksize_col - 1) * (rate_col - 1))) // stride_col + 1
        else:
            out_row = (in_row - 1) // stride_row + 1
            out_col = (in_col - 1) // stride_col + 1

        out_shape = [out_batch, out_row, out_col, out_depth]
        return out_shape

    def infer_dtype(self, input_x):
        """infer dtype"""
        validator.check_tensor_type_same({"input_x": input_x}, mstype.number_type, self.name)
        return input_x


class EmbeddingLookup(PrimitiveWithInfer):
    """
    Returns a slice of input tensor based on the specified indices.

    This Primitive has the similar functionality as GatherV2 operating on `axis = 0`, but has three more inputs:
    `offset`, `reduce_scatter_flag` and `split_num`. This primitive runs on the host instead of devices.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The Tensor slice, instead of the entire Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of `input_params`,
          and the exceeding part will be filled with 0 in the output.
        - **offset** (int) - Specifies the offset value of this `input_params` slice. Thus the real indices
          are equal to `input_indices` minus `offset`.
        - **reduce_scatter_flag** (bool) - Specifies whether perform reduce_scatter on host or not.
          Only constant value is allowed.
        - **split_num** (int) - Specifies the number of partitions of the reduce_scatter produces. This variable
          is used only if `reduce_scatter_flag` is True. Only constant value is allowed.


    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([[5, 2], [8, 5]]), mindspore.int32)
        >>> offset = 4
        >>> reduce_scatter_flag = False
        >>> split_num = 1
        >>> out = P.EmbeddingLookup()(input_params, input_indices, offset, reduce_scatter_flag, split_num)
        [[[10, 11], [0 ,0]], [[0, 0], [10, 11]]]
    """
    @prim_attr_register
    def __init__(self):
        """init index_select"""
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=['params', 'indices', 'offset', 'reduce_scatter_flag', 'split_num'],
                                outputs=['output'])
        self.add_prim_attr('primitive_target', 'CPU')

    def __infer__(self, params, indices, offset, reduce_scatter_flag=False, split_num=2):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_type_same({"indices": indices['dtype']}, mstype.int_type, self.name)
        validator.check_subclass("offset", offset['dtype'], mstype.int_, self.name)
        validator.check_subclass("split_num", split_num['dtype'], mstype.int_, self.name)
        if split_num['value'] < 1:
            raise ValueError("The parameter 'split_num' must be positive, but got %d." % split_num)
        params_shp = params['shape']
        out_shape = indices['shape'] + params_shp[1:]
        if reduce_scatter_flag is None:
            raise ValueError("The value of 'reduce_scatter_flag' is None.")
        reduce_scatter_flag_value = reduce_scatter_flag['value']
        if split_num is None:
            raise ValueError("The value of 'split_num_value' is None.")
        split_num_value = split_num['value']
        if reduce_scatter_flag_value is True:
            # Partition the tensor along the dimension 0. The shape size of dimension 0 should be divisible by
            # (split_num * 8)
            if out_shape[0] % (split_num_value * 8) != 0:
                raise ValueError("The dimension 0 of the shape: %d, is not divisible by: %d." %
                                 (out_shape[0], (split_num_value * 8)))
            # After 'Concat' on host, the shape size of dimension 0 is: out_shape[0] // 8
            out_shape[0] = out_shape[0] // 8
        out = {'shape': out_shape,
               'dtype': params['dtype'],
               'value': None}
        return out
