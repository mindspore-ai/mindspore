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
from mindspore import log as logger
from mindspore._c_expression import security
from mindspore._c_expression import Tensor as Tensor_
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops.primitive import prim_attr_register, Primitive, PrimitiveWithInfer


def _check_mode(class_name):
    """Check for PyNative mode."""
    mode = context.get_context('mode')
    if mode == context.PYNATIVE_MODE:
        raise RuntimeError(f"For '{class_name}', the operator does not support PyNative mode.")


def _check_summary_param(name, value, class_name):
    """Checks the name and value is valid for summary."""
    _check_mode(class_name)
    n_type = name['dtype']
    n_value = name['value']
    validator.check_value_type('name', n_type, [type(mstype.string)], class_name)
    if not n_value:
        raise ValueError(f"For '{class_name}', the name must be valid string, but got '{n_value}'.")

    v_type = value['dtype']
    validator.check_value_type('value', v_type, [type(mstype.tensor)], class_name)


# Note: The return value of the summary operator is not used,
# so there's nothing special about the return `dtype` or `shape`, any value is ok.
# The `value` should be set to None, else summary operators may be optimized at compile graph phase,
# it cause summary operators can not record data in constant folding scene.
SUMMARY_RETURN_VALUE = {'dtype': mstype.int32, 'shape': [1], 'value': None}


class ScalarSummary(Primitive):
    """
    This operator will put a scalar to a summary file with protocol buffer format. It must be used with SummaryRecord
    or SummaryCollector, which specify the directory of the summary file. The summary file can
    be loaded and shown by MindInsight, see `MindInsight documents <https://www.mindspore.cn/
    mindinsight/docs/en/r2.0.0-alpha/index.html>`_ for details.

    Inputs:
        - **name** (str) - The name of the input variable, it must not be an empty string.
        - **value** (Tensor) - The value of scalar, and the dim of `value` must be 0 or 1.

    Raises:
        TypeError: If `name` is not a str.
        TypeError: If `value` is not a Tensor.
        ValueError: If dim of `value` is greater than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor, set_context
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
        >>> set_context(mode=mindspore.GRAPH_MODE)
        >>> summary = SummaryDemo()(Tensor(3), Tensor(4))
        >>> print(summary)
        7
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScalarSummary."""

        if security.enable_security():
            raise ValueError('The Summary is not supported, please without `-s on` and recompile source.')

        self.add_prim_attr("side_effect_io", True)

    def __call__(self, *args):
        raise RuntimeError(f"PyNative not support Operator '{self.__class__.__name__}'")


class ImageSummary(PrimitiveWithInfer):
    """
    This operator will put an image tensor to a summary file with protocol buffer format. It must be used with
    SummaryRecord or SummaryCollector, which specify the directory of the summary file. The summary file can
    be loaded and shown by MindInsight, see `MindInsight documents <https://www.mindspore.cn/
    mindinsight/docs/en/r2.0.0-alpha/index.html>`_ for details.

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
        ...         self.summary(name, x)
        ...         return x
        ...
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ImageSummary."""

        if security.enable_security():
            raise ValueError('The Summary is not supported, please without `-s on` and recompile source.')

        self.add_prim_attr("side_effect_io", True)

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        # The shape dim of image should be 4.
        v_shape = value['shape']
        image_dim = 4
        if len(v_shape) != image_dim:
            raise ValueError(f"For '{self.name}', the dimension of 'value' must be {image_dim},"
                             f" but got {len(v_shape)}.")

        return SUMMARY_RETURN_VALUE

    def __call__(self, *args):
        raise RuntimeError(f"PyNative not support Operator '{self.__class__.__name__}'")


class TensorSummary(Primitive):
    """
    This operator will put a tensor to a summary file with protocol buffer format. It must be used with SummaryRecord
    or SummaryCollector, which specify the directory of the summary file. The summary file can
    be loaded and shown by MindInsight, see `MindInsight documents <https://www.mindspore.cn/
    mindinsight/docs/en/r2.0.0-alpha/index.html>`_ for details.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of tensor, and the rank of tensor must be greater than 0.

    Raises:
        TypeError: If `name` is not a str.
        TypeError: If `value` is not a Tensor.
        ValueError: If rank of `value` is 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor, set_context
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
        >>> set_context(mode=mindspore.GRAPH_MODE)
        >>> summary = SummaryDemo()(Tensor([[1]]), Tensor([[2]]))
        >>> print(summary)
        [[3]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TensorSummary."""

        if security.enable_security():
            raise ValueError('The Summary is not supported, please without `-s on` and recompile source.')

        self.add_prim_attr("side_effect_io", True)

    def __call__(self, *args):
        raise RuntimeError(f"PyNative not support Operator '{self.__class__.__name__}'")


class HistogramSummary(PrimitiveWithInfer):
    """
    This operator will calculate the histogram of a tensor and put it to a summary file with protocol buffer format.
    It must be used with SummaryRecord or SummaryCollector, which specify the directory of the summary file.
    The summary file can be loaded and shown by MindInsight, see `MindInsight documents <https://www.mindspore.cn/
    mindinsight/docs/en/r2.0.0-alpha/index.html>`_ for details.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of tensor, and the rank of tensor must be greater than 0.

    Raises:
        TypeError: If `name` is not a str.
        TypeError: If `value` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor, set_context
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
        >>> set_context(mode=mindspore.GRAPH_MODE)
        >>> summary = SummaryDemo()(Tensor([1, 2]), Tensor([3, 4]))
        >>> print(summary)
        [4 6]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize HistogramSummary."""

        if security.enable_security():
            raise ValueError('The Summary is not supported, please without `-s on` and recompile source.')

        self.add_prim_attr("side_effect_io", True)

    def __infer__(self, name, value):
        _check_summary_param(name, value, self.__class__.__name__)

        v_shape = value['shape']
        # In the summary, the histogram value should be a tensor whose shape is not [].
        if not v_shape:
            raise ValueError(f"For '{self.name}', the type of 'value' must be tensor, "
                             f"its shape should not be [], but got {v_shape}.")

        return SUMMARY_RETURN_VALUE

    def __call__(self, *args):
        raise RuntimeError(f"PyNative not support Operator '{self.__class__.__name__}'")


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
        TypeError: If `f` is not a function of MindSpore.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, jit
        >>> a = Tensor(np.array([1.0]).astype(np.float32))
        >>> b = Tensor(np.array([0.2]).astype(np.float32))
        >>> def clip_gradient(dx):
        ...     ret = dx
        ...     if ret > a:
        ...         ret = a
        ...
        ...     if ret < b:
        ...         ret = b
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
        ...     @jit
        ...     def f(x, y):
        ...         return clip_test(x, y)
        ...
        ...     def fd(x, y):
        ...         return grad_all(clip_test)(x, y)
        ...
        ...     print("forward: ", f(Tensor(np.array([1.1]).astype(np.float32)),
        ...         Tensor(np.array([0.1]).astype(np.float32))))
        ...     print("clip_gradient:", fd(Tensor(np.array([1.1]).astype(np.float32)),
        ...         Tensor(np.array([0.1]).astype(np.float32))))
        >>> InsertGradientOfClipDemo()
        forward: [0.11000001]
        clip_gradient: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000003e-01]),
                        Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
    """

    @prim_attr_register
    def __init__(self, f):
        """Initialize InsertGradientOf."""
        self.add_prim_attr('side_effect_backprop', True)
        self.f = f

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        return x_type


class HookBackward(PrimitiveWithInfer):
    """
    This operation is used as a tag to hook gradient in intermediate variables. Note that this function
    is only supported in pynative mode.

    Note:
        The hook function must be defined like `hook_fn(grad) -> new gradient or None`, where the 'grad' is the
        gradient passed to the primitive. The 'grad' may be modified by returning a new gradient and passed to next
        primitive. The difference between a hook function and callback of InsertGradientOf is that the hook function is
        executed in the python environment while callback will be parsed and added to the graph.

    Args:
        hook_fn (Function): Python function. hook function.
        cell_id (str, optional): Used to identify whether the function registered by the hook is actually registered on
                       the specified cell object. For example, 'nn.Conv2d' is a cell object.
                       The default value of `cell_id` is empty string(""), in this case, the system will automatically
                       register a value of `cell_id`.
                       The value of `cell_id` currently does not support custom values.

    Inputs:
        - **input** (Tensor) - The variable to hook.

    Outputs:
        - **output** (Tensor) - Returns `input` directly. `HookBackward` does not affect the forward result.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If `hook_fn` is not a function of python.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> from mindspore.ops import GradOperation
        >>> ms.set_context(mode=ms.PYNATIVE_MODE)
        >>> def hook_fn(grad):
        ...     print(grad)
        ...
        >>> hook = ops.HookBackward(hook_fn)
        >>> def hook_test(x, y):
        ...     z = x * y
        ...     z = hook(z)
        ...     z = z * y
        ...     return z
        ...
        >>> grad_all = GradOperation(get_all=True)
        >>> def backward(x, y):
        ...     return grad_all(hook_test)(x, y)
        ...
        >>> output = backward(Tensor(1, ms.float32), Tensor(2, ms.float32))
        (Tensor(shape=[], dtype=Float32, value= 2),)
        >>> print(output)
        (Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 4))
    """

    def __init__(self, hook_fn, cell_id=""):
        """Initialize HookBackward."""
        super(HookBackward, self).__init__(self.__class__.__name__)
        if not isinstance(hook_fn, (FunctionType, MethodType)):
            raise TypeError(f"For '{self.name}', the type of 'hook_fn' must be python function, "
                            f"but got {type(hook_fn)}.")
        if cell_id != "":
            logger.warning(f"The args 'cell_id' of HookBackward will be removed in a future version. If the value of "
                           f"'cell_id' is set, the hook function will not work.")
        self.add_prim_attr("cell_id", cell_id)
        self.init_attrs["cell_id"] = cell_id
        self.cell_id = cell_id
        self.add_backward_hook_fn(hook_fn)

    def infer_shape(self, *inputs_shape):
        if len(inputs_shape) == 1:
            return inputs_shape[0]
        return inputs_shape

    def infer_dtype(self, *inputs_type):
        for dtype in inputs_type:
            validator.check_subclass("input", dtype, [mstype.tensor], self.name)
        if len(inputs_type) == 1:
            return inputs_type[0]
        return inputs_type


class Print(Primitive):
    """
    Print the inputs to stdout.

    Refer to :func:`mindspore.ops.print_` for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        Tensor(shape=[2, 1], dtype=Int32, value=
        [[1]
         [1]])
        Tensor(shape=[2, 2], dtype=Int32, value=
        [[1 1]
         [1 1]])
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Print."""
        if security.enable_security():
            raise ValueError(
                'The Print is not supported, please without `-s on` and recompile source.')
        self.add_prim_attr("side_effect_io", True)

    def __call__(self, *args):
        for arg in args:
            if isinstance(arg, Parameter):
                print(Tensor_.__repr__(arg))
            elif isinstance(arg, Tensor_):
                print(arg.__repr__())
            else:
                print(arg)


class Assert(PrimitiveWithInfer):
    """
    Asserts whether the given condition is True.
    If input condition is identified to be false, print a list of the tensor in data.

    Args:
        summarize (int, optional): The number of entries to be printed in each tensor while the given condition is
            identified to be False. Default: 3.

    Inputs:
        - **condition** (Union[Tensor[bool], bool]) - The condition to be identified.
        - **input_data** (Union[tuple[Tensor], list[Tensor]]) - The tensors to be printed out when the condition
          is false.

    Raises:
        TypeError: If `summarize` is not an int.
        TypeError: If `condition` is neither a Tensor nor a bool.
        TypeError: If `input_data` is neither a tuple nor a list.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([-1, 0, 1, 2, 3]).astype(np.int32))
        >>> b = Tensor(np.array([1, 2, 3, 4, 5]).astype(np.float32))
        >>> assert1 = ops.Assert(3)
        >>> assert1(False, [a, b])
        For 'Assert' condition is false.
        input data: [-1 0 1]
        input data: [1 2 3]
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "mindspore/ops/primitive.py", line 294, in __call__
            return _run_op(self, self.name, args)
          File "mindspore/common/api.py", line 99, in wrapper
            results = fn(*arg, **kwargs)
          File "mindspore/ops/primitive.py", line 743, in _run_op
            output = real_run_op(obj, op_name, args)
        RuntimeError: assert failed
    """

    @prim_attr_register
    def __init__(self, summarize=3):
        """Initialize Assert"""
        if security.enable_security():
            raise ValueError(
                'The Assert is not supported, please without `-s on` and recompile source.')
        self.add_prim_attr("side_effect_io", True)
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
