# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Custom operator"""
import os
import inspect
import json
import functools
from mindspore import ops
from mindspore.ops.op_info_register import RegOp
from mindspore._c_expression import Oplib


class CustomRegOp(RegOp):
    """Class for Custom op info register"""

    def __init__(self, op_name="Custom"):
        super(CustomRegOp, self).__init__(op_name)
        self.target_ = "UnKnown"

    def input(self, index=None, name=None, param_type="required", **kwargs):
        """
        Register Custom op input information.

        Args:
            index (int): Order of the input. Default: None.
            name (str): Name of the input. Default: None.
            param_type (str): Param type of the input. Default: None.
            kwargs (dict): Other information of the input.
        """
        param_list = [index, name, param_type]
        key_list = ["index", "name", "param_type"]
        fn_list = [self._is_int, self._is_string, self._is_string]
        input_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.inputs.append(input_dict)
        return self

    def output(self, index=None, name=None, param_type="required", **kwargs):
        """
        Register Custom op output information.

        Args:
            index (int): Order of the output. Default: None.
            name (str): Name of the output. Default: None.
            param_type (str): Param type of the output. Default: None.
            kwargs (dict): Other information of the output.
        """
        param_list = [index, name, param_type]
        key_list = ["index", "name", "param_type"]
        fn_list = [self._is_int, self._is_string, self._is_string]
        output_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.outputs.append(output_dict)
        return self

    def attr(self, name=None, param_type=None, value_type=None, default_value=None, **kwargs):
        """
        Register Custom op attribute information.

        Args:
            name (str): Name of the attribute. Default: None.
            param_type (str): Param type of the attribute. Default: None.
            value_type (str): Value type of the attribute. Default: None.
            default_value (str): Default value of attribute. Default: None.
            kwargs (dict): Other information of the attribute.
        """
        param_list = [name, param_type, value_type, default_value]
        key_list = ["name", "param_type", "type", "default_value"]
        fn_list = [self._is_string]
        attr_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.attr_.append(attr_dict)
        return self

    def target(self, target=None):
        """
        Register Custom op target information.

        Args:
            target (str): Device target for current operator information, should be one of ["Ascend", "GPU", "CPU"].
                Please note that target and the `func_type` of `Custom` op have some constraints.
                If func_type is "akg", target can be one of ["Ascend", "GPU"].
                If func_type is "tbe", target can only be "Ascend".
                If func_type is "lib", target can only be "GPU".
                If func_type is "py_func", target can only be "CPU".
                Default: None.
        """
        self._is_string(target)
        self.target_ = target
        return self


def custom_op_info_register(*reg_info):
    r"""
    A decorator which is used to bind the registration information to the `func` parameter of `Custom` op.

    Note:
        The 'reg_info' will be added into oplib.

    Args:
        reg_info (tuple): Each item represents registration information in json format.

    Returns:
        Function, returns a decorator for op info register.
    """

    def decorator(func):
        setattr(func, "reg_info", reg_info)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class Custom(ops.PrimitiveWithInfer):
    r"""
    `Custom` primitive is used for user defined operators and is to enhance the expressive ability of built-in
    primitives. You can construct a `Custom` object with a predefined function, which describes the computation
    logic of a user defined operator. You can also construct another `Custom` object with another predefined
    function if needed. Then these `Custom` objects can be directly used in neural networks.

    .. warning::
        This is an experimental prototype that is subject to change.

    Args:
        func (Union[function, str]): If func is of function type, then func should be a Python function which describes
            the computation logic of a user defined operator. The function can be one of the following:
            1. A AKG operator implementation function, which can use ir builder/tvm compute/hybrid grammar.
            2. A TBE operator implementation function.
        out_shape (Union[function, list, tuple]): The output shape infer function or the value of output shape of func.
            If func has single output, then the value of output shape is a list.
            If func has multiple outputs, then the value of output shape is a tuple of list, each list represents the
            shape of each output.
        out_dtype (Union[function, :class:`mindspore.dtype`, tuple[:class:`mindspore.dtype`]]): The output dtype infer
            function or the value of output dtype of func.
            If func has single output, then the value of output shape is a mindspore.dtype.
            If func has multiple outputs, then the value of output shape is a tuple of mindspore.dtype.
        func_type (str): The implementation type of func, should be one of ["akg", "tbe", "lib", "py_func"].
        grad (function): The gradient function of func. Default: None.
        reg_info (Union[str, dict, list, tuple]): Represents the registration information of func with json format of
            type str or dict. The registration information specifies supported formats of input and output, attributes
            and target of func. If reg_info is a list or tuple, then each item should be with json format of type str
            or dict, which represents the registration information of func in a specific target. You need to invoke
            `CustomRegOp` or the subclass of `RegOp` to generate the registration info for func. Then you can invoke
            `custom_op_info_register` to bind the reg info to func or just pass the reg info to `reg_info` parameter.
            The `reg_info` parameter takes higher priority then `custom_op_info_register`. Default: None.

    Inputs:
        - **input** (Union(tuple, list)) - The input tuple or list is made up of multiple tensors, and attributes
          value(optional).

    Outputs:
        tuple[Tensor], execution results.

    Raises:
        TypeError: If the type of `func` is invalid or the type of register information for `func` is invalid.
        ValueError: If the register information is invalid.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops.op_info_register import DataType
        >>> from mindspore.nn import Cell
        >>>
        >>> square_with_bias_op_info = CustomRegOp() \
        >>>     .fusion_type("OPAQUE") \
        >>>     .attr("bias", "required", "float") \
        >>>     .input(0, "x") \
        >>>     .output(0, "y") \
        >>>     .dtype_format(DataType.F32_Default, DataType.F32_Default) \
        >>>     .dtype_format(DataType.F16_Default, DataType.F16_Default) \
        >>>     .target("Ascend") \
        >>>     .get_op_info()
        >>>
        >>> @custom_op_info_register(square_with_bias_op_info)
        >>> def square_with_bias(input_x, output_y, bias=0.0, kernel_name="square_with_bias"):
        >>>     import te.lang.cce
        >>>     from te import tvm
        >>>     from topi import generic
        >>>     from topi.cce import util
        >>>
        >>>     shape = input_x.get("shape")
        >>>     dtype = input_x.get("dtype").lower()
        >>>
        >>>     shape = util.shape_refine(shape)
        >>>     data = tvm.placeholder(shape, name="data", dtype=dtype.lower())
        >>>
        >>>     with tvm.target.cce():
        >>>         res0 = te.lang.cce.vmul(data, data)
        >>>         res = te.lang.cce.vadds(res0, bias)
        >>>         sch = generic.auto_schedule(res)
        >>>
        >>>     config = {"print_ir": False,
        >>>               "name": kernel_name,
        >>>               "tensor_list": [data, res]}
        >>>
        >>>     te.lang.cce.cce_build_code(sch, config)
        >>>
        >>> class Net(Cell):
        >>>     def __init__(self):
        >>>         super(Net1, self).__init__()
        >>>         self.square_with_bias = Custom(square_with_bias, out_shape=[2, 3], out_dtype=mstype.float32, \
        >>>                                        func_type="tbe")
        >>>
        >>>     def construct(self, x):
        >>>         res = self.square_with_bias(x, 1.0)
        >>>         return res
    """

    registered_func = {}

    def __init__(self, func, out_shape, out_dtype, func_type, grad=None, reg_info=None):
        ops.PrimitiveWithInfer.__init__(self, "Custom")

        self.supported_targets = ["Ascend", "GPU", "CPU"]
        self.func = func
        self.func_name = ""
        if callable(self.func):
            # Get the original function if func is decorated
            if "__wrapped__" in self.func.__dict__:
                self.func = self.func.__dict__["__wrapped__"]
            self.func_name = self.func.__name__
        elif isinstance(self.func, str):
            self.func_name = self.func
        else:
            raise TypeError("func should be of type function or str, but got {}".format(type(self.func)))
        self.add_prim_attr("func_name", self.func_name)
        self.out_shape = out_shape
        self.out_dtype = out_dtype
        self.grad = grad
        self.func_type = func_type
        # Register info
        self.register_info(reg_info)

        if func_type == "akg":
            func_source_str = inspect.getsource(self.func)
            index = func_source_str.find("def ")
            if index != -1:
                func_source_str = func_source_str[index:]
            self.add_prim_attr('func_source_str', func_source_str)
            if "ir_builder" in func_source_str:
                self.func_type = "ir_builder"
            elif "compute" in func_source_str:
                self.func_type = "tvm_compute"
            else:
                self.func_type = "hybrid"
        self.add_prim_attr("func_type", self.func_type)

    def infer_shape(self, *args):
        if callable(self.out_shape):
            return self.out_shape(*args)
        return self.out_shape

    def infer_dtype(self, *args):
        if callable(self.out_dtype):
            return self.out_dtype(*args)
        return self.out_dtype

    def get_bprop(self):
        return self.grad

    def register_info(self, info):
        """Register reg_info."""
        reg_info = info
        if reg_info is None and hasattr(self.func, "reg_info"):
            reg_info = getattr(self.func, "reg_info")
        reg_info_list = self.get_expanded_list(reg_info)
        already_add_attr = False
        for reg_info in reg_info_list:
            if not isinstance(reg_info, (str, dict)):
                continue
            if isinstance(reg_info, str):
                reg_info = json.loads(reg_info)
            target = self.get_target(reg_info)
            # Reg info for func is only registered once for a certain target
            if self.has_registered(target):
                continue
            # Register
            reg_info = self.reformat_reg_info(reg_info, target)
            reg_info_str = json.dumps(reg_info)
            if isinstance(self.func, str):
                imply_path = self.func
            else:
                imply_path = os.path.realpath(inspect.getfile(self.func))
            op_lib = Oplib()
            if not op_lib.reg_op(reg_info_str, imply_path):
                raise ValueError('Invalid reg info {}: {}\n'.format(imply_path, reg_info_str))
            # Add inputs name to attr
            if not already_add_attr:
                self.add_inputs_name_to_attr(reg_info)
                already_add_attr = True
            self.save_register_status(target)

    def get_expanded_list(self, data):
        """Recursive function to parse elements in list or tuple."""
        data_list = []
        if isinstance(data, (list, tuple)):
            for i in data:
                tmp_list = self.get_expanded_list(i)
                for ii in tmp_list:
                    data_list.append(ii)
        else:
            data_list.append(data)
        return data_list

    def has_registered(self, target):
        """Check if registration information is registered in target."""
        if callable(self.func) and target in getattr(self.func, "registered_targets", []):
            return True
        if isinstance(self.func, str) and target in Custom.registered_func.get(self.func, []):
            return True
        return False

    def save_register_status(self, target):
        """Save registration status for target."""
        if callable(self.func):
            registered_targets = getattr(self.func, "registered_targets", [])
            registered_targets.append(target)
            setattr(self.func, "registered_targets", registered_targets)
        elif isinstance(self, str):
            if isinstance(Custom.registered_func.get(self.func), list):
                Custom.registered_func[self.func].append(target)
            else:
                Custom.registered_func[self.func] = [target]

    def reformat_reg_info(self, reg_info, target):
        """Reformat registration information."""
        if not isinstance(reg_info, dict):
            raise TypeError("reg_info should be of type dict, but got {}".format(type(reg_info)))
        reg_info["op_name"] = self.func_name
        reg_info["imply_type"] = self.get_imply_type(reg_info, target)
        # Supplement necessary info for TBE if these information is missing in reg_info
        if reg_info["imply_type"] == "TBE":
            if reg_info.get("attr") is not None and isinstance(reg_info["attr"], list):
                for i, item in enumerate(reg_info["attr"]):
                    if isinstance(item, dict) and item.get("value") is None:
                        reg_info["attr"][i]["value"] = "all"
            reg_info["async_flag"] = reg_info.get("async_flag", False)
            reg_info["binfile_name"] = self.func_name + ".so"
            reg_info["compute_cost"] = reg_info.get("compute_cost", 10)
            reg_info["kernel_name"] = self.func_name
            reg_info["partial_flag"] = reg_info.get("partial_flag", True)
            reg_info["need_check_supported"] = reg_info.get("need_check_supported", False)
        # Supplement necessary info for AKG if these information is missing in reg_info
        if reg_info["imply_type"] == "AKG":
            target_to_processor = {"Ascend": "AiCore", "GPU": "CUDA", "CPU": "CPU"}
            reg_info["processor"] = reg_info.get("processor", target_to_processor.get(target))
        return reg_info

    def get_target(self, reg_info):
        """Get target information."""
        target = None
        if isinstance(reg_info, dict):
            # Get target from reg_info["target"]
            target = reg_info.get("target")
            # Infer target from reg_info["processor"], reg_info generated from AkgGpuRegOp or AkgAscendRegOp
            #   will have the processor information.
            if target not in self.supported_targets:
                processor_to_target = {"AiCore": "Ascend", "CUDA": "GPU", "CPU": "CPU"}
                target = processor_to_target.get(reg_info.get("processor"))
            # Infer target from reg_info["imply_type"]
            if target not in self.supported_targets:
                imply_type_to_target = {"TBE": "Ascend", "GPU": "GPU", "CPU": "CPU"}
                target = imply_type_to_target.get(reg_info.get("imply_type"))
        # Infer target from func_type
        if target not in self.supported_targets:
            func_type_to_target = {"tbe": "Ascend"}
            target = func_type_to_target.get(self.func_type)
        if target not in self.supported_targets:
            raise ValueError("target should be one of {}, but got {}".format(self.supported_targets, target))
        return target

    def get_imply_type(self, reg_info, target):
        """Get imply_typ information."""
        # Get imply_type from reg_info["imply_type"]
        if isinstance(reg_info, dict) and isinstance(reg_info.get("imply_type"), str) and \
                reg_info["imply_type"].strip():
            return reg_info["imply_type"]
        # Infer imply_type from func_type
        func_type_to_imply_type = {"akg": "AKG", "tbe": "TBE", "lib": target, "py_func": target}
        return func_type_to_imply_type.get(self.func_type, "AKG")

    def add_inputs_name_to_attr(self, reg_info):
        """Save inputs name to primitive's attr."""
        if not isinstance(reg_info, dict):
            return
        tensor_inputs = reg_info.get("inputs", [])
        attr = reg_info.get("attr", [])
        input_names = []
        attr_names = []
        for item in tensor_inputs:
            if isinstance(item, dict) and item.get("name") is not None:
                input_names.append(item["name"])
        for item in attr:
            if isinstance(item, dict) and item.get("name") is not None:
                input_names.append(item["name"])
                attr_names.append(item["name"])
        # input_names include tensor input names and attr names
        self.add_prim_attr("input_names", input_names)
        self.add_prim_attr("attr_names", attr_names)
