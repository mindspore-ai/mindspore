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

"""Operators info register."""

import os
import json
import inspect
from mindspore._c_expression import Oplib
from mindspore._checkparam import ParamValidator as validator

# path of built-in op info register.
BUILT_IN_OPS_REGISTER_PATH = "mindspore/ops/_op_impl"


def op_info_register(op_info):
    """
    A decorator used as register of operator implementation.

    Note:
        'op_info' must be a str of json format represent the op info, the op info will be added into oplib.

    Args:
        op_info (str or dict): op info of json format.

    Returns:
        Function, returns a decorator for op info register.
    """
    def register_decorator(func):
        if isinstance(op_info, dict):
            op_info_real = json.dumps(op_info)
        else:
            op_info_real = op_info
        validator.check_type("op_info", op_info_real, [str])
        op_lib = Oplib()
        file_path = os.path.realpath(inspect.getfile(func))
        # keep the path custom ops implementation.
        imply_path = "" if BUILT_IN_OPS_REGISTER_PATH in file_path else file_path
        if not op_lib.reg_op(op_info_real, imply_path):
            raise ValueError('Invalid op info {}:\n{}\n'.format(file_path, op_info_real))

        def wrapped_function(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped_function
    return register_decorator


class RegOp():
    """
    Base class for op info register.

    Args:
        op_name (str): Name of op.
        inputs (list): Inputs inoformation of the op.
        outputs (list): Outputs information of the op.
        attr_ (list): Attribute information of the op.
        dtype_format_ (list): Dtype and format information of the op.
    """

    def __init__(self, op_name=""):
        if not isinstance(op_name, str):
            raise ValueError("op name value must be string")
        if not op_name.strip():
            raise ValueError("op name is empty")
        self.op_name = op_name
        self.inputs = []
        self.outputs = []
        self.attr_ = []
        self.dtype_format_ = []

    def is_string(self, value):
        """
        Check if the value is a str type.

        Args:
            value: Parameter to to check.

        Raises:
            TypeError: If the type of value is not a str.
        """
        if not isinstance(value, str):
            raise TypeError("%s value must be str" % str(value))

    def is_int(self, value):
        """
        Check if the value is a int.

        Args:
            value: Parameter to to check.

        Raises:
            TypeError: If the type of value is not a int.
        """
        if not isinstance(value, int):
            raise TypeError("%s value must be int" % str(value))

    def is_bool(self, value):
        """
        Check if the value is a bool.

        Args:
            value: Parameter to to check.

        Raises:
            TypeError: If the type of value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError("%s value must be bool" % str(value))

    def dtype_format(self, *args):
        """
        Register dtype and format.

        Args:
            args (tuple): Value of dtype and format.

        Raises:
            ValueError: If the size of args not equal to input size add output size.
            TypeError: If the type of args is not tuple.
        """
        if len(self.inputs) + len(self.outputs) != len(args):
            raise ValueError("input size add output size must be equal to detype format size")
        dtype_format = []
        for arg in args:
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise ValueError("dtype and format value must be tuple of two elements")
            self.is_string(arg[0])
            self.is_string(arg[1])
            dtype_format.append(arg)
        self.dtype_format_.append(tuple(dtype_format))
        return self

    def get_op_info(self):
        """
        Return all registration information for this instance.

        The '_' character ending the key is removed here for compatibility with previous version.

        Key will be unified into an underlined form later.
        """
        op_info = {}
        for key, value in self.__dict__.items():
            if isinstance(key, str) and key.endswith('_'):
                op_info[key.rstrip('_')] = value
            else:
                op_info[key] = value
        return op_info


class TBERegOp(RegOp):
    """Class for TBE op info register."""

    def __init__(self, op_name=""):
        super(TBERegOp, self).__init__(op_name)
        self.imply_type = "TBE"
        self.fusion_type_ = ''
        self.async_flag_ = False
        self.binfile_name_ = ''
        self.compute_cost_ = 10
        self.kernel_name_ = ''
        self.partial_flag_ = False
        self.reshape_type_ = ''
        self.dynamic_format_ = False
        self.op_pattern_ = ""

    def fusion_type(self, fusion_type):
        """
        Register fusion type.

        Args:
            fusion_type (str): Value of fusion type.
        """
        self.is_string(fusion_type)
        self.fusion_type_ = fusion_type
        return self

    def async_flag(self, async_flag):
        """
        Register async flag.

        Args:
            async_flag (bool): Value of async flag.
        """
        self.is_bool(async_flag)
        self.async_flag_ = async_flag
        return self

    def binfile_name(self, binfile_name):
        """
        Register binfile name.

        Args:
            binfile_name (str): Name of op binfile.
        """
        self.is_string(binfile_name)
        self.binfile_name_ = binfile_name
        return self

    def compute_cost(self, compute_cost):
        """
        Register compute cost.

        Args:
            compute_cost (int): Value of compute cost.
        """
        self.is_int(compute_cost)
        self.compute_cost_ = compute_cost
        return self

    def kernel_name(self, kernel_name):
        """
        Register kernel name.

        Args:
            kernel_name (str): Name of op kernel.
        """
        self.is_string(kernel_name)
        self.kernel_name_ = kernel_name
        return self

    def partial_flag(self, partial_flag):
        """
        Register partial flag.

        Args:
            partial_flag (bool): Value of partial flag.
        """
        self.is_bool(partial_flag)
        self.partial_flag_ = partial_flag
        return self

    def reshape_type(self, reshape_type):
        """
        Register reshape type.

        Args:
            reshape_type (str): Value of reshape type.
        """
        self.is_string(reshape_type)
        self.reshape_type_ = reshape_type
        return self

    def dynamic_format(self, dynamic_format):
        """
        Register dynamic format.

        Args:
            reshape_type (bool): Value of dynamic format.
        """
        self.is_bool(dynamic_format)
        self.dynamic_format_ = dynamic_format
        return self

    def op_pattern(self, pattern=None):
        """
        Register op pattern information.

        Args:
            pattern (str): Value of op pattern.
        """
        if pattern is not None and self.istring(pattern):
            self.op_pattern_ = pattern
        return self

    def attr(self, name=None, param_type=None, value_type=None, value=None, default_value=None, **kwargs):
        """
        Register op attribute information.

        Args:
            name (str): Name of the attribute. Default: None.
            param_type (str): Param type of the attribute. Default: None.
            type (str): Type of the attribute. Default: None.
            value (str): Value of the attribute. Default: None.
            default_value (str): Default value of attribute. Default: None.
            kwargs (dict): Other information for the attribute.
        """
        param_list = [name, param_type, value_type, value, default_value]
        attr_dict = {}
        for index, element in enumerate(param_list):
            if element is not None:
                self.is_string(element)
                if index == 0:
                    attr_dict["name"] = element
                elif index == 1:
                    attr_dict["param_type"] = element
                elif index == 2:
                    attr_dict["type"] = element
                elif index == 3:
                    attr_dict["value"] = element
                elif index == 4:
                    attr_dict["default_value"] = element
        if kwargs:
            attr_dict = dict(attr_dict, **kwargs)
        self.attr_.append(attr_dict)
        return self

    def input(self, index=None, name=None, need_compile=None, param_type=None, shape=None, **kwargs):
        """
        Register op input information.

        Args:
            index (int): Order of the input. Default: None.
            name (str): Name of the input. Default: None.
            need_compile (bool): The input need compile whether or not. Default: None.
            param_type (str): Type of the input. Default: None.
            shape (str): Shape of the input. Default: None.
            kwargs (dict): Other information for the input.
        """
        param_list = [index, name, need_compile, param_type, shape]
        input_dict = {}
        for idx, element in enumerate(param_list):
            if element is not None:
                if idx == 0:
                    self.is_int(element)
                    input_dict["index"] = element
                elif idx == 1:
                    self.is_string(element)
                    input_dict["name"] = element
                elif idx == 2:
                    self.is_bool(element)
                    input_dict["need_compile"] = element
                elif idx == 3:
                    self.is_string(element)
                    input_dict["param_type"] = element
                elif idx == 4:
                    self.is_string(element)
                    input_dict["shape"] = element
        if kwargs:
            input_dict = dict(input_dict, **kwargs)
        self.inputs.append(input_dict)
        return self

    def output(self, index=None, name=None, need_compile=None, param_type=None, shape=None, **kwargs):
        """
        Register op output information.

        Args:
            index (int): Order of the output. Default: None.
            name (str): Name of the output. Default: None.
            need_compile (bool): The output need compile whether or not. Default: None.
            param_type (str): Type of the output. Default: None.
            shape (str): Shape of the output. Default: None.
            kwargs (dict): Other information for the output.
        """
        param_list = [index, name, need_compile, param_type, shape]
        output_dict = {}
        for idx, element in enumerate(param_list):
            if element is not None:
                if idx == 0:
                    self.is_int(element)
                    output_dict["index"] = element
                elif idx == 1:
                    self.is_string(element)
                    output_dict["name"] = element
                elif idx == 2:
                    self.is_bool(element)
                    output_dict["need_compile"] = element
                elif idx == 3:
                    self.is_string(element)
                    output_dict["param_type"] = element
                elif idx == 4:
                    self.is_string(element)
                    output_dict["shape"] = element
        if kwargs:
            output_dict = dict(output_dict, **kwargs)
        self.outputs.append(output_dict)
        return self

class DataType():
    """
    Various combinations of dtype and formatself.

    The current list below maybe not completed. If necessary, please add it.
    """

    BOOL_None = ("bool", "")
    BOOL_Default = ("bool", "DefaultFormat")
    BOOL_5HD = ("bool", "NC1HWC0")
    BOOL_NCHW = ("bool", "NCHW")
    BOOL_NHWC = ("bool", "NHWC")
    BOOL_HWCN = ("bool", "HWCN")

    I8_None = ("int8", "")
    I8_Default = ("int8", "DefaultFormat")
    I8_5HD = ("int8", "NC1HWC0")
    I8_FracZ = ("int8", "Fracz")
    I8_FracNZ = ("int8", "FRACTAL_NZ")
    I8_NCHW = ("int8", "NCHW")
    I8_NHWC = ("int8", "NHWC")
    I8_HWCN = ("int8", "HWCN")

    U8_None = ("uint8", "")
    U8_Default = ("uint8", "DefaultFormat")
    U8_5HD = ("uint8", "NC1HWC0")
    U8_FracZ = ("uint8", "Fracz")
    U8_FracNZ = ("uint8", "FRACTAL_NZ")
    U8_NCHW = ("uint8", "NCHW")
    U8_NHWC = ("uint8", "NHWC")
    U8_HWCN = ("uint8", "HWCN")

    I16_None = ("int16", "")
    I16_Default = ("int16", "DefaultFormat")
    I16_5HD = ("int16", "NC1HWC0")
    I16_FracZ = ("int16", "Fracz")
    I16_FracNZ = ("int16", "FRACTAL_NZ")
    I16_NCHW = ("int16", "NCHW")
    I16_NHWC = ("int16", "NHWC")
    I16_HWCN = ("int16", "HWCN")

    U16_None = ("uint16", "")
    U16_Default = ("uint16", "DefaultFormat")
    U16_5HD = ("uint16", "NC1HWC0")
    U16_FracZ = ("uint16", "Fracz")
    U16_FracNZ = ("uint16", "FRACTAL_NZ")
    U16_NCHW = ("uint16", "NCHW")
    U16_NHWC = ("uint16", "NHWC")
    U16_HWCN = ("uint16", "HWCN")

    I32_None = ("int32", "")
    I32_Default = ("int32", "DefaultFormat")
    I32_5HD = ("int32", "NC1HWC0")
    I32_FracZ = ("int32", "Fracz")
    I32_FracNZ = ("int32", "FRACTAL_NZ")
    I32_NCHW = ("int32", "NCHW")
    I32_NHWC = ("int32", "NHWC")
    I32_HWCN = ("int32", "HWCN")

    U32_None = ("uint32", "")
    U32_Default = ("uint32", "DefaultFormat")
    U32_5HD = ("uint32", "NC1HWC0")
    U32_FracZ = ("uint32", "Fracz")
    U32_FracNZ = ("uint32", "FRACTAL_NZ")
    U32_NCHW = ("uint32", "NCHW")
    U32_NHWC = ("uint32", "NHWC")
    U32_HWCN = ("uint32", "HWCN")

    I64_None = ("int64", "")
    I64_Default = ("int64", "DefaultFormat")
    I64_5HD = ("int64", "NC1HWC0")
    I64_FracZ = ("int64", "Fracz")
    I64_FracNZ = ("int64", "FRACTAL_NZ")
    I64_NCHW = ("int64", "NCHW")
    I64_NHWC = ("int64", "NHWC")
    I64_HWCN = ("int64", "HWCN")

    U64_None = ("uint64", "")
    U64_Default = ("uint64", "DefaultFormat")
    U64_5HD = ("uint64", "NC1HWC0")
    U64_FracZ = ("uint64", "Fracz")
    U64_FracNZ = ("uint64", "FRACTAL_NZ")
    U64_NCHW = ("uint64", "NCHW")
    U64_NHWC = ("uint64", "NHWC")
    U64_HWCN = ("uint64", "HWCN")

    F16_None = ("float16", "")
    F16_Default = ("float16", "DefaultFormat")
    F16_5HD = ("float16", "NC1HWC0")
    F16_FracZ = ("float16", "Fracz")
    F16_FracNZ = ("float16", "FRACTAL_NZ")
    F16_C1HWNCoC0 = ("float16", "C1HWNCoC0")
    F16_NCHW = ("float16", "NCHW")
    F16_NHWC = ("float16", "NHWC")
    F16_HWCN = ("float16", "HWCN")

    F32_None = ("float32", "")
    F32_Default = ("float32", "DefaultFormat")
    F32_5HD = ("float32", "NC1HWC0")
    F32_FracZ = ("float32", "Fracz")
    F32_FracNZ = ("float32", "FRACTAL_NZ")
    F32_C1HWNCoC0 = ("float32", "C1HWNCoC0")
    F32_NCHW = ("float32", "NCHW")
    F32_NHWC = ("float32", "NHWC")
    F32_HWCN = ("float32", "HWCN")
