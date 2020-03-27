# Copyright 2019 Huawei Technologies Co., Ltd
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

"""dsl create helping function"""
import akg
from akg.utils import format_transform as ft_util

class TensorUtils:
    """Class for creating tensor."""
    CREATE_SCH_ONLY = 'create_sch_only'

    @classmethod
    def get_tensor_attrs(cls, tensor):
        """get tensor attrs."""
        tensor_attrs = dict()
        if "attrs" in dir(tensor.op):
            tensor_attrs = dict(tensor.op.attrs.items())
        return tensor_attrs

    @classmethod
    def update_tensor_attrs(cls, tensor, attrs):
        """update tensor attrs."""
        tensor_attrs = cls.get_tensor_attrs(tensor)
        tensor_attrs.update(attrs)
        tensor = akg.tvm.compute(tensor.shape,
                                 lambda *indice: tensor[indice],
                                 name=tensor.op.name,
                                 tag=tensor.op.tag,
                                 attrs=tensor_attrs)
        return tensor

    @classmethod
    def is_create_sch_only(cls, tensor):
        tensor_attrs = cls.get_tensor_attrs(tensor)
        if cls.CREATE_SCH_ONLY in tensor_attrs.keys():
            return True
        return False

    @classmethod
    def is_output_value(cls, tensor):
        """check output value."""
        return not cls.is_create_sch_only(tensor)

    @classmethod
    def inplace_set(cls, input_tensor, output_tensor, buffer_name="data_buf"):
        """inplace set."""
        input_tensor_shape = ft_util.get_shape(input_tensor)
        output_tensor_shape = ft_util.get_shape(output_tensor)
        if not input_tensor_shape == output_tensor_shape:
            raise RuntimeError("Shape of the input_tensor and the output_tensor should be equal, "
                               "but got %s and %s"%(input_tensor_shape, output_tensor_shape))
        output_tensor = cls.update_tensor_attrs(output_tensor, {cls.CREATE_SCH_ONLY: 1})
        data_buf = akg.tvm.decl_buffer(input_tensor.shape, input_tensor.dtype, name=buffer_name)
        binds_info = {input_tensor: data_buf, output_tensor: data_buf}
        return output_tensor, binds_info

    @classmethod
    def inplace_set_tensors(cls, input_tensors, output_tensors, buffer_names=None):
        """
        inplace set for tensors

        Args:
            in_tensors (Union[list, tuple]): Origin input tensors.
            out_tensors (Union[list, tuple]): Origin output tensors.
            buffer_names (Union[list, tuple] or None): Buffer names used to bind.

        Return:
            inplace_tensors (list): Output tensors with the inplace info.
            binds_infos (dict): Dictionary that maps the input tensor and the output
                                tensor to buffer.
        """
        if not buffer_names:
            buffer_names = ["data_buf_%s" % i for i in range(len(input_tensors))]
        for arg in (input_tensors, output_tensors, buffer_names):
            if not isinstance(arg, (tuple, list)):
                raise RuntimeError("arg must be tuple or list!")
        if len(input_tensors) != len(output_tensors) or len(input_tensors) != len(buffer_names):
            raise RuntimeError("length of the input_tensors, output_tensors and buffer_names must be equal!")

        inplace_tensors = []
        binds_infos = dict()
        for input_tensor, output_tensor, buffer_name in zip(input_tensors, output_tensors, buffer_names):
            inplace_tensor, binds_info = cls.inplace_set(input_tensor, output_tensor, buffer_name)
            inplace_tensors.append(inplace_tensor)
            binds_infos.update(binds_info)
        return inplace_tensors, binds_infos

def produce_shapes(shape1, shape2):
    """two input shapes produce three output shape."""
    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    out_shape = []
    for i in range(output_shape_len):
        if (shape1[i] != shape2[i]) and (shape1[i] != 1) and (shape2[i] != 1):
            raise RuntimeError("input shapes not match!")
        out_shape.append(shape1[i] if shape1[i] > shape2[i] else shape2[i])

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape
