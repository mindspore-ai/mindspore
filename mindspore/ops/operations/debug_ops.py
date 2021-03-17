# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""debug_ops"""
from types import FunctionType, MethodType

from mindspore import context
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ..primitive import prim_attr_register, PrimitiveWithInfer


def _check_mode(class_name):
    """Check for PyNative mode."""
    mode = context.get_context('mode')
    if mode == context.PYNATIVE_MODE:
        raise RuntimeError(f'{class_name} operator does not support PyNative mode.')


def _check_summary_param(name, value, class_name):
    """Checks the name and value is valid for summary."""
    _check_mode(class_name)
    n_type = name['dtype']
    n_value = name['value']
    validator.check_value_type('name', n_type, [type(mstype.string)], class_name)
    if not n_value:
        raise ValueError(f"For 'name' the value should by valid string in {class_name}, but got an empty string.")

    v_type = value['dtype']
    validator.check_value_type('value', v_type, [type(mstype.tensor)], class_name)


# Note: The return value of the summary operator is not used,
# so there's nothing special about the return `dtype` or `shape`, any value is ok.
# The `value` should be set to None, else summary operators may be optimized at compile graph phase,
# it cause summary operators can not record data in constant folding scene.
SUMMARY_RETURN_VALUE = {'dtype': mstype.int32, 'shape': [1], 'value': None}


class ScalarSummary(PrimitiveWithInfer):
    """
    Outputs a scalar to a protocol buffer through a scalar summary operator.

    Inputs:
        - **name** (str) - The name of the input variable, it must not be an empty string.
        - **value** (Tensor) - The value of scalar, and the shape of value must be [] or [1].

    Raises:
        TypeError: If `name` is not a str.
        TypeError: If `value` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>>
        >>>
        >>> class SummaryDemo(nn.Cell):
        ...     def __init__(self,):
        ...         super(SummaryDemo, self).__init__()
        ...         self.summary = ops.ScalarSummary()
        ...         self.add = ops.Add()
        ...
        ...     def construct(self, x, y):
        ...         name = "x"
        ...         self.summary(name, x)
        ...         x = self.add(x, y)
        ...         return x
        ...
    """

    @prim_attr_register
    def __init__(self):
        """init"""
        self.add_prim_attr("side_effect_io", True)

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        v_shape = value['shape']
        # In the summary, the value whose shape is [1] is also considered as a scalar.
        if v_shape and v_shape != [1]:
            raise ValueError(f"For 'value' the type should be scalar, "
                             f"shape should be [] or [1] in {self.__class__.__name__}, but got {v_shape}.")

        return SUMMARY_RETURN_VALUE


class ImageSummary(PrimitiveWithInfer):
    """
    Outputs the image tensor to protocol buffer through image summary operator.

    Inputs:
        - **name** (str) - The name of the input variable, it must not be an empty string.
        - **value** (Tensor) - The value of image, the rank of tensor must be 4.

    Raises:
        TypeError: If `name` is not a str.
        TypeError: If `value` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:

        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>>
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.summary = ops.ImageSummary()
        ...
        ...     def construct(self, x):
        ...         name = "image"
        ...         out = self.summary(name, x)
        ...         return out
        ...
    """

    @prim_attr_register
    def __init__(self):
        """init"""
        self.add_prim_attr("side_effect_io", True)

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        # The shape dim of image should be 4.
        v_shape = value['shape']
        image_dim = 4
        if len(v_shape) != image_dim:
            raise ValueError(f"For 'value' the dim should be {image_dim} in {self.__class__.__name__},"
                             f" but got {len(v_shape)}.")

        return SUMMARY_RETURN_VALUE


class TensorSummary(PrimitiveWithInfer):
    """
    Outputs a tensor to a protocol buffer through a tensor summary operator.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of tensor, and the rank of tensor must be greater than 0.

    Raises:
        TypeError: If `name` is not a str.
        TypeError: If `value` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>>
        >>>
        >>> class SummaryDemo(nn.Cell):
        ...     def __init__(self,):
        ...         super(SummaryDemo, self).__init__()
        ...         self.summary = ops.TensorSummary()
        ...         self.add = ops.Add()
        ...
        ...     def construct(self, x, y):
        ...         x = self.add(x, y)
        ...         name = "x"
        ...         self.summary(name, x)
        ...         return x
        ...
    """

    @prim_attr_register
    def __init__(self):
        """init"""
        self.add_prim_attr("side_effect_io", True)

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        v_shape = value['shape']
        # In the summary, the value whose shape is [] is not considered as a tensor.
        if not v_shape:
            raise ValueError(f"For 'value' the type should be tensor in {self.__class__.__name__}, "
                             f"shape should not be [].")

        return SUMMARY_RETURN_VALUE


class HistogramSummary(PrimitiveWithInfer):
    """
    Outputs the tensor to protocol buffer through histogram summary operator.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of tensor, and the rank of tensor must be greater than 0.

    Raises:
        TypeError: If `name` is not a str.
        TypeError: If `value` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>>
        >>>
        >>> class SummaryDemo(nn.Cell):
        ...     def __init__(self,):
        ...         super(SummaryDemo, self).__init__()
        ...         self.summary = ops.HistogramSummary()
        ...         self.add = ops.Add()
        ...
        ...     def construct(self, x, y):
        ...         x = self.add(x, y)
        ...         name = "x"
        ...         self.summary(name, x)
        ...         return x
        ...
    """

    @prim_attr_register
    def __init__(self):
        """init"""
        self.add_prim_attr("side_effect_io", True)

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        v_shape = value['shape']
        # In the summary, the histogram value should be a tensor whose shape is not [].
        if not v_shape:
            raise ValueError(f"For 'value' the type should be tensor in {self.__class__.__name__}, "
                             f"shape should not be [].")

        return SUMMARY_RETURN_VALUE


class InsertGradientOf(PrimitiveWithInfer):
    """
    Attaches callback to the graph node that will be invoked on the node's gradient.

    Args:
        f (Function): MindSpore's Function. Callback function.

    Inputs:
        - **input_x** (Any) - The graph node to attach to.

    Outputs:
        Tensor, returns `input_x` directly. `InsertGradientOf` does not affect the forward result.

    Raises:
        TypeError: If `f` is not a function of mindspore.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> def clip_gradient(dx):
        ...     ret = dx
        ...     if ret > 1.0:
        ...         ret = 1.0
        ...
        ...     if ret < 0.2:
        ...         ret = 0.2
        ...
        ...     return ret
        ...
        >>> clip = ops.InsertGradientOf(clip_gradient)
        >>> grad_all = ops.GradOperation(get_all=True)
        >>> def InsertGradientOfClipDemo():
        ...     def clip_test(x, y):
        ...         x = clip(x)
        ...         y = clip(y)
        ...         c = x * y
        ...         return c
        ...
        ...     @ms_function
        ...     def f(x, y):
        ...         return clip_test(x, y)
        ...
        ...     def fd(x, y):
        ...         return grad_all(clip_test)(x, y)
        ...
        ...     print("forward: ", f(1.1, 0.1))
        ...     print("clip_gradient:", fd(1.1, 0.1))
        ...
    """

    @prim_attr_register
    def __init__(self, f):
        self.add_prim_attr('side_effect_backprop', True)
        self.f = f

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        return x_type


class HookBackward(PrimitiveWithInfer):
    """
    This operation is used as a tag to hook gradient in intermediate variables.  Note that this function
    is only supported in Pynative Mode.

    Note:
        The hook function must be defined like `hook_fn(grad) -> Tensor or None`,
        where grad is the gradient passed to the primitive and gradient may be
        modified and passed to next primitive. The difference between a hook function and
        callback of InsertGradientOf is that a hook function is executed in the python
        environment while callback will be parsed and added to the graph.

    Args:
        hook_fn (Function): Python function. hook function.

    Inputs:
        - **inputs** (Tensor) - The variable to hook.

    Raises:
        TypeError: If `inputs` are not a Tensor.
        TypeError: If `hook_fn` is not a function of python.

    Examples:
        >>> def hook_fn(grad_out):
        ...     print(grad_out)
        ...
        >>> grad_all = GradOperation(get_all=True)
        >>> hook = ops.HookBackward(hook_fn)
        >>> def hook_test(x, y):
        ...     z = x * y
        ...     z = hook(z)
        ...     z = z * y
        ...     return z
        ...
        >>> def backward(x, y):
        ...     return grad_all(hook_test)(x, y)
        ...
        >>> output = backward(1, 2)
        >>> print(output)
    """

    def __init__(self, hook_fn, cell_id=""):
        super(HookBackward, self).__init__(self.__class__.__name__)
        self.add_prim_attr("cell_id", cell_id)
        self.init_attrs["cell_id"] = cell_id
        if not isinstance(hook_fn, (FunctionType, MethodType)):
            raise TypeError("Hook function should be python function type.")
        self.register_hook(hook_fn)
        self.cell_id = cell_id

    def infer_shape(self, *inputs_shape):
        if len(inputs_shape) == 1:
            return inputs_shape[0]
        return inputs_shape

    def infer_dtype(self, *inputs_type):
        if len(inputs_type) == 1:
            return inputs_type[0]
        return inputs_type


class Print(PrimitiveWithInfer):
    """
    Outputs the tensor or string to stdout.

    Note:
        In pynative mode, please use python print function.
        In graph mode, the bool, int and float would be converted into Tensor to print,
        str remains unchanged.

    Inputs:
        - **input_x** (Union[Tensor, bool, int, float, str]) - The graph node to attach to.
          Supports multiple inputs which are separated by ','.

    Raises:
        TypeError: If `input_x` is not one of the following: Tensor, bool, int, float, str.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> class PrintDemo(nn.Cell):
        ...     def __init__(self):
        ...         super(PrintDemo, self).__init__()
        ...         self.print = ops.Print()
        ...
        ...     def construct(self, x, y):
        ...         self.print('Print Tensor x and Tensor y:', x, y)
        ...         return x
        ...
        >>> x = Tensor(np.ones([2, 1]).astype(np.int32))
        >>> y = Tensor(np.ones([2, 2]).astype(np.int32))
        >>> net = PrintDemo()
        >>> result = net(x, y)
        Print Tensor x and Tensor y:
        [[1]
         [1]]
        [[1 1]
         [1 1]]
    """

    @prim_attr_register
    def __init__(self):
        self.add_prim_attr("side_effect_io", True)

    def __call__(self, *args):
        for arg in args:
            print(arg)

    def infer_shape(self, *inputs):
        return [1]

    def infer_dtype(self, *inputs):
        # check argument types except the last one (io state).
        for ele in inputs[:-1]:
            validator.check_subclass("input", ele,
                                     [mstype.tensor, mstype.int_, mstype.float_, mstype.bool_, mstype.string],
                                     self.name)
        return mstype.int32


class Assert(PrimitiveWithInfer):
    """
    Asserts that the given condition is True.
    If input condition evaluates to false, print the list of tensor in data.

    Args:
        summarize (int): Print this many entries of each tensor.

    Inputs:
        - **condition** [Union[Tensor[bool], bool]] - The condition to evaluate.
        - **input_data** (Union(tuple[Tensor], list[Tensor])) - The tensors to print out when condition is false.

    Raises:
        TypeError: If `summarize` is not an int.
        TypeError: If `condition` is neither a Tensor nor a bool.
        TypeError: If `input_data` is neither a tuple nor a list.

    Examples:
        >>> class AssertDemo(nn.Cell):
        ...     def __init__(self):
        ...         super(AssertDemo, self).__init__()
        ...         self.assert1 = ops.Assert(summarize=10)
        ...         self.add = ops.Add()
        ...
        ...     def construct(self, x, y):
        ...         data = self.add(x, y)
        ...         self.assert1(True, [data])
        ...         return data
        ...
    """

    @prim_attr_register
    def __init__(self, summarize=3):
        """Initialize Assert"""
        self.summarize = validator.check_value_type("summarize", summarize, [int], self.name)

    def infer_shape(self, condition, inputs):
        condition_len = len(condition)
        validator.check_int(condition_len, 1, Rel.LE, "condition's rank", self.name)
        if condition_len == 1:
            validator.check_equal_int(condition[0], 1, "condition[0]", self.name)
        return [1]

    def infer_dtype(self, condition, inputs):
        validator.check_scalar_or_tensor_types_same({"condition": condition}, [mstype.bool_], self.name)
        for dtype in inputs:
            validator.check_subclass("input", dtype, [mstype.tensor], self.name)
        return mstype.int32
