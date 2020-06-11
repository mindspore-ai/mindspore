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

"""debug_ops"""
from types import FunctionType, MethodType
from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import prim_attr_register, PrimitiveWithInfer, Primitive


def _check_summary_param(name, value, class_name):
    """Check the name and value is valid for summary."""
    n_type = name['dtype']
    n_value = name['value']
    validator.check_value_type('name', n_type, [type(mstype.string)], class_name)
    if not n_value:
        raise ValueError(f"For 'name' the value should by valid string in {class_name}, but got an empty string.")

    v_type = value['dtype']
    validator.check_value_type('value', v_type, [type(mstype.tensor)], class_name)


class ScalarSummary(PrimitiveWithInfer):
    """
    Output scalar to protocol buffer through scalar summary operator.

    Inputs:
        - **name** (str) - The name of the input variable, it should not be an empty string.
        - **value** (Tensor) - The value of scalar, and the shape of value should be [] or [1].

    Examples:
        >>> class SummaryDemo(nn.Cell):
        >>>     def __init__(self,):
        >>>         super(SummaryDemo, self).__init__()
        >>>         self.summary = P.ScalarSummary()
        >>>         self.add = P.TensorAdd()
        >>>
        >>>     def construct(self, x, y):
        >>>         name = "x"
        >>>         self.summary(name, x)
        >>>         x = self.add(x, y)
        >>>         return x
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        v_shape = value['shape']
        # In the summary, the value whose shape is [1] is also considered as a scalar.
        if v_shape and v_shape != [1]:
            raise ValueError(f"For 'value' the type should be scalar, "
                             f"shape should be [] or [1] in {self.__class__.__name__}, but got {v_shape}.")

        return value


class ImageSummary(PrimitiveWithInfer):
    """
    Output image tensor to protocol buffer through image summary operator.

    Inputs:
        - **name** (str) - The name of the input variable, it should not be an empty string.
        - **value** (Tensor) - The value of image, the rank of tensor should be 4.

    Examples:
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.summary = P.ImageSummary()
        >>>
        >>>     def construct(self, x):
        >>>         name = "image"
        >>>         out = self.summary(name, x)
        >>>         return out
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        # The shape dim of image should be 4.
        v_shape = value['shape']
        image_dim = 4
        if len(v_shape) != image_dim:
            raise ValueError(f"For 'value' the dim should be {image_dim} in {self.__class__.__name__},"
                             f" but got {len(v_shape)}.")

        return value


class TensorSummary(PrimitiveWithInfer):
    """
    Output tensor to protocol buffer through tensor summary operator.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of tensor, and the rank of tensor should be greater than 0.

    Examples:
        >>> class SummaryDemo(nn.Cell):
        >>>     def __init__(self,):
        >>>         super(SummaryDemo, self).__init__()
        >>>         self.summary = P.TensorSummary()
        >>>         self.add = P.TensorAdd()
        >>>
        >>>     def construct(self, x, y):
        >>>         x = self.add(x, y)
        >>>         name = "x"
        >>>         self.summary(name, x)
        >>>         return x
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        v_shape = value['shape']
        # In the summary, the value whose shape is [] is not considered as a tensor.
        if not v_shape:
            raise ValueError(f"For 'value' the type should be tensor in {self.__class__.__name__}, "
                             f"shape should not be [].")

        return value


class HistogramSummary(PrimitiveWithInfer):
    """
    Output tensor to protocol buffer through histogram summary operator.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of tensor, and the rank of tensor should be greater than 0.

    Examples:
        >>> class SummaryDemo(nn.Cell):
        >>>     def __init__(self,):
        >>>         super(SummaryDemo, self).__init__()
        >>>         self.summary = P.HistogramSummary()
        >>>         self.add = P.TensorAdd()
        >>>
        >>>     def construct(self, x, y):
        >>>         x = self.add(x, y)
        >>>         name = "x"
        >>>         self.summary(name, x)
        >>>         return x
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        v_shape = value['shape']
        # In the summary, the histogram value should be a tensor whose shape is not [].
        if not v_shape:
            raise ValueError(f"For 'value' the type should be tensor in {self.__class__.__name__}, "
                             f"shape should not be [].")

        return value


class InsertGradientOf(PrimitiveWithInfer):
    """
    Attach callback to graph node that will be invoked on the node's gradient.

    Args:
        f (Function): MindSpore's Function. Callback function.

    Inputs:
        - **input_x** (Any) - The graph node to attach to.

    Outputs:
        Tensor, returns `input_x` directly. `InsertGradientOf` does not affect the forward result.

    Examples:
        >>> def clip_gradient(dx):
        >>>     ret = dx
        >>>     if ret > 1.0:
        >>>         ret = 1.0
        >>>
        >>>     if ret < 0.2:
        >>>         ret = 0.2
        >>>
        >>>     return ret
        >>>
        >>> clip = P.InsertGradientOf(clip_gradient)
        >>> grad_all = C.GradOperation('get_all', get_all=True)
        >>> def InsertGradientOfClipDemo():
        >>>     def clip_test(x, y):
        >>>         x = clip(x)
        >>>         y = clip(y)
        >>>         c = x * y
        >>>         return c
        >>>
        >>>     @ms_function
        >>>     def f(x, y):
        >>>         return clip_test(x, y)
        >>>
        >>>     def fd(x, y):
        >>>         return grad_all(clip_test)(x, y)
        >>>
        >>>     print("forward: ", f(1.1, 0.1))
        >>>     print("clip_gradient:", fd(1.1, 0.1))
    """

    @prim_attr_register
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        """run in PyNative mode."""
        return x

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        return x_type


class HookBackward(PrimitiveWithInfer):
    """
    Used as tag to hook gradient in intermediate variables.

    Note:
        The hook function should be defined like `hook_fn(grad) -> Tensor or None`,
        which grad is the gradient passed to the primitive and gradient may be
        modified and passed to nex primitive. the difference between hook function and
        callback of InsertGradientOf is that hook function is executed in python
        environment while callback will be parsed and added to the graph.

    Args:
        hook_fn (Function): Python function. hook function.

    Inputs:
        - **inputs** (Tensor) - The variable to hook.

    Examples:
        >>> def hook_fn(grad_out):
        >>>     print(grad_out)
        >>>
        >>> hook = P.HookBackward(hook_fn)
        >>>
        >>> def hook_test(x, y):
        >>>     z = x * y
        >>>     z = hook(z)
        >>>     z = z * y
        >>>     return z
        >>>
        >>> def backward(x, y):
        >>>     return C.grad_all(hook_test)(x, y)
        >>>
        >>> backward(1, 2)
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
    Output tensor or string to stdout.

    Note:
        The print operation cannot support the following cases currently.

        1. The type of tensor is float64 or bool.

        2. The data of tensor is a scalar type.

        In pynative mode, please use python print function.

    Inputs:
        - **input_x** (Union[Tensor, str]) - The graph node to attach to. The input supports
          multiple strings and tensors which are separated by ','.

    Examples:
        >>> class PrintDemo(nn.Cell):
        >>>     def __init__(self):
        >>>         super(PrintDemo, self).__init__()
        >>>         self.print = P.Print()
        >>>
        >>>     def construct(self, x, y):
        >>>         self.print('Print Tensor x and Tensor y:', x, y)
        >>>         return x
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, *args):
        for arg in args:
            print(arg)

    def infer_shape(self, *inputs):
        return [1]

    def infer_dtype(self, *inputs):
        for dtype in inputs:
            validator.check_subclass("input", dtype, (mstype.tensor, mstype.string), self.name)
        return mstype.int32


class Debug(Primitive):
    """
    Print tensor value.

    Inputs:
        - **value** (Tensor) - The value of tensor.

    Examples:
        >>> class DebugNN(nn.Cell):
        >>>     def __init__(self,):
        >>>         self.debug = nn.Debug()
        >>>
        >>>     def construct(self, x, y):
        >>>         x = self.add(x, y)
        >>>         self.debug(x)
        >>>         return x
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, *args, **kwargs):
        pass
