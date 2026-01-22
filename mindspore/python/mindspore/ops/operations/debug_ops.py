# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import inspect
from mindspore import log as logger
from mindspore._c_expression import security, HookType
from mindspore._c_expression import TensorPy as Tensor_
from mindspore._c_expression import _tensordump_exec
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.graph.jit_context import jit_context
from mindspore.ops.primitive import prim_attr_register, Primitive, PrimitiveWithInfer
from mindspore._checkparam import check_hook_fn
from mindspore.ops import operations as P

SUMMARY_TENSOR_CACHE = []


def _cache_summary_data(op_name, define_name, tensor):
    """Cache summary tensor data."""
    global SUMMARY_TENSOR_CACHE
    SUMMARY_TENSOR_CACHE.append([op_name, define_name, tensor])


def _check_summary_param(name, value, class_name):
    """Checks the name and value is valid for summary."""
    n_type = name['dtype']
    n_value = name['value']
    validator.check_value_type('name', n_type, [type(mstype.string)], class_name)
    if not n_value:
        raise ValueError(f"For '{class_name}', the name must be valid string, but got '{n_value}'.")

    v_type = value['dtype']
    validator.check_value_type('value', v_type, [type(mstype.tensor_type)], class_name)


# Note: The return value of the summary operator is not used,
# so there's nothing special about the return `dtype` or `shape`, any value is ok.
# The `value` should be set to None, else summary operators may be optimized at compile graph phase,
# it cause summary operators can not record data in constant folding scene.
SUMMARY_RETURN_VALUE = {'dtype': mstype.int32, 'shape': [1], 'value': None}


class ScalarSummary(Primitive):
    """
    This operator will put a scalar to a summary file with protocol buffer format.
    It must be used with :class:`mindspore.SummaryRecord` or :class:`mindspore.SummaryCollector`,
    which specify the directory of the summary file.
    In Ascend platform with graph mode, the environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
    can be set to solve operator execution failure when calling this operator intensively.

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
        >>> from mindspore import ops
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
        self.add_prim_attr("channel_name", "ms_scalar_summary")
        self.add_prim_attr("dyn_input_sizes", [-1, 1])

    def __call__(self, *args):
        _cache_summary_data(self.name, args[0], args[1])


class ImageSummary(Primitive):
    """
    This operator will put an image tensor to a summary file with protocol buffer format. It must be used with
    SummaryRecord or SummaryCollector, which specify the directory of the summary file.
    In Ascend platform with graph mode, the environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
    can be set to solve execution failure when calling this operator intensively.

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
        >>> from mindspore import ops
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
        self.add_prim_attr("channel_name", "ms_image_summary")
        self.add_prim_attr("dyn_input_sizes", [-1, 1])

    def __call__(self, *args):
        _cache_summary_data(self.name, args[0], args[1])


class TensorSummary(Primitive):
    """
    This operator will put a tensor to a summary file with protocol buffer format. It must be used with SummaryRecord
    or SummaryCollector, which specify the directory of the summary file.
    In Ascend platform with graph mode, the environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
    can be set to solve operator execution failure when calling this operator intensively.

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
        >>> from mindspore import ops
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
        self.add_prim_attr("channel_name", "ms_tensor_summary")

    def __call__(self, *args):
        _cache_summary_data(self.name, args[0], args[1])


class TensorDump(Primitive):
    """
    Save the Tensor as an npy file in numpy format.

    .. warning::
        The parameter input_output will no longer support the value 'all'.

    .. note::
        In Ascend platform with graph mode, the environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
        can be set to solve operator execution failure when outputting big tensor or outputting tensor intensively.

    Args:
        input_output (str, optional): Used to control Tensordump behavior.
            Available value is one of ['in', 'out']. Default value is ``out``.

            In case of OpA --> RedistributionOps --> OpB,
            The dump data of OpA's output is not equal to OpB's input (Due to the redistribution operators).
            So the parameter input_output is to handle this situation.

            Assuming OpA's output is used as both Tensordump's input parameter and OpB's input parameter.
            Different requirements of saving dump data can be achieved by configuring parameter input_output:

            - If the input_output is 'out', the dump data contains only OpA's output slice.
            - If the input_output is 'in', the dump data contains only OpB's input slice.

            For input_output is 'in', the input slice npy file format is:
            fileName_dumpMode_dtype_id.npy.

            For input_output is 'out', the output slice npy file format is:
            fileName_dtype_id.npy.

            - fileName: Value of the parameter file
              (if parameter file_name is a user-specified path, the value of fileName is the last level of the path).
            - dumpMode: Value of the parameter input_output.
            - dtype: The original data type. Data of type bfloat16 stored in the .npy file will be converted to float32.
            - id: An auto increment ID.

    Inputs:
        - **file** (str) - The path of the file to be saved.
        - **input_x** (Tensor) - Input Tensor of any dimension.

    Raises:
        TypeError: If `file` is not a str.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import time
        >>> from mindspore import nn, Tensor, ops
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_device(device_target="Ascend")
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.dump = ops.TensorDump()
        ...
        ...     def construct(self, x):
        ...         x += 1.
        ...         self.dump('add', x)
        ...         x /= 2.
        ...         self.dump('div', x)
        ...         x *= 5.
        ...         self.dump('mul', x)
        ...         return x
        ...
        >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
        >>> input_x = Tensor(x)
        >>> net = Net()
        >>> out = net(input_x)
        >>> time.sleep(0.5)
        >>> add = np.load('add_float32_0.npy')
        >>> print(add)
        [[2. 3. 4. 5.]
         [6. 7. 8. 9.]]
    """
    @prim_attr_register
    def __init__(self, input_output='out'):
        """Initialize TensorDump."""
        if security.enable_security():
            raise ValueError('The TensorDump is not supported, please without `-s on` and recompile source.')
        if input_output not in ['in', 'out']:
            raise ValueError(f"The 'input_output' argument should be one of ['in', 'out'], but got: {input_output}")
        self.add_prim_attr("side_effect_io", True)
        self.add_prim_attr("channel_name", "ms_tensor_dump")

    def __call__(self, file, input_x):
        validator.check_value_type('file', file, [str], self.__class__.__name__)
        if not file:
            raise ValueError("For 'TensorDump', the input argument[file] cannot be an empty string.")
        validator.check_value_type('input_x', input_x, [Tensor], self.__class__.__name__)
        _tensordump_exec(file, input_x)


class HistogramSummary(Primitive):
    """
    This operator will calculate the histogram of a tensor and put it to a summary file with protocol buffer format.
    It must be used with SummaryRecord or SummaryCollector, which specify the directory of the summary file.
    In Ascend platform with graph mode, the environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
    can be set to solve operator execution failure when calling this operator intensively.

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
        >>> from mindspore import ops
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
        self.add_prim_attr("channel_name", "ms_histogram_summary")
        self.add_prim_attr("dyn_input_sizes", [-1, 1])

    def __call__(self, *args):
        _cache_summary_data(self.name, args[0], args[1])


class InsertGradientOf(Primitive):
    """
    Attaches callback to the graph node that will be invoked on the node's gradient.

    .. warning::
        In the callback, exercise caution when using side-effect operators,
        such as the TensorDump operator, as current support is incomplete.

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


class DumpGradient(Primitive):
    """
        The `DumpGradient` Primitive is a hook, used to dump dout which pass to `x`.

        Inputs:
            - **path** (str) - The path of the file to be saved.
            - **x** (Tensor) - Input Tensor of any dimension.
            - **input_output** (str) - support value should be one of ['in', 'out'].

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import ops
            >>> from mindspore import Tensor
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> ms.set_device(device_target="Ascend")
            >>> dg = ops.DumpGradient()
            >>> def dout_dump_test(x, y):
            ...     x = dg("x_dout.npy", x, 'out')
            ...     print(f"x value is {x}")
            ...     z = x * y
            ...     return z
            >>> ms_grad = ms.grad(dout_dump_test, grad_position=(0,1))
            >>> x_grad, y_grad = ms_grad(Tensor(1, ms.float32), Tensor(2, ms.float32))
            >>> print(f"x grad is {x_grad}, y_grad is {y_grad}")
            >>> x_grad_npy = np.load("x_dout.npy")
            >>> print(f"load x_grad from npy, x_grad is {x_grad_npy}")
            x value is 1.0
            x grad is 2.0, y grad is 1.0
            load x_grad from npy, x_grad is array(2., dtype=float32)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, path, x, input_output):
        def _dump_hook(dout):
            P.TensorDump()(path, dout)
            return dout
        x = P.InsertGradientOf(_dump_hook)(x)
        return x


class Morph(PrimitiveWithInfer):
    """
    The `Morph` Primitive is used to encapsulate a user-defined function `fn`, allowing it to be used as a custom
    Primitive.

    The `Morph` Primitive is primarily designed for custom graph optimization in GRAPH mode. For example, it supports
    encapsulation of irregular collective communications (such as :func:`mindspore.ops.AlltoAllV`) in distributed
    auto-parallel training scenarios.

    When the `Morph` Primitive is applied to inputs, it is actually the encapsulated user-defined function `fn` that is
    applied to the inputs.

    The main difference between the `Morph` Primitive and :func:`mindspore.ops.Custom` is that the former is expanded
    and replaced by the user-defined `fn` before automatic differentiation, so there is no need to implement a backward
    function.

    .. note::
        - This primitive is only supported in GRAPH_MODE.
        - A user-defined bprop (by argument: `bprop_fn`) is allowed for `Morph`.
        - `fn` and `bprop_fn` must satisfy the syntax constraints of the graph mode.
        - `vararg`, `kwarg`, `kwonlyargs` and free variables are not supported in user-defined function.

    Args:
        fn (Function): MindSpore's function, user-defined function.
        infer_shape (Function): MindSpore's function, user-defined infer_shape function.
        infer_dtype (Function): MindSpore's function, user-defined infer_dtype function.
        bprop_fn (Function, optional): MindSpore's function, user-defined bprop function, default: ``None``.

    Inputs:
        The inputs of user-defined `fn`.

    Outputs:
        The outputs of user-defined `fn`.

    Raises:
        RuntimeError: if not used in GRAPH_MODE.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import context, nn, ops, Tensor, Parameter
        >>>
        >>> np_weight0 = np.array([1.0, 2.0, 3.0])
        >>> np_weight1 = np.array([4.0, 5.0, 6.0])
        >>> np_input_x = np.array([7.0, 8.0, 9.0])
        >>>
        >>> def infer_dtype(args):
        ...     return args
        >>>
        >>> def infer_shape(args):
        ...     return args
        >>>
        >>> def mul_by(*args):
        ...     def inner(x):
        ...         return args[0] * x
        ...     return inner
        >>>
        >>> NUMBER_100 = 100
        >>> class MorphNet(nn.Cell):
        ...     def __init__(self):
        ...         super(MorphNet, self).__init__()
        ...         self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        ...         self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
        ...         self.mul_by_100 = ops.Morph(mul_by(NUMBER_100), infer_shape, infer_dtype)
        ...     def construct(self, x):
        ...         a = x * self.weight0
        ...         b = self.mul_by_100(a)
        ...         out = b * self.weight1
        ...         return out
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> input_x = Tensor(np_input_x, ms.float32)
        >>> net = MorphNet()
        >>> grad_op = ops.GradOperation(get_all=True, get_by_list=True)
        >>> grad_net = grad_op(net, net.trainable_params())
        >>> bwd_out = grad_net(input_x)
        >>> x_grad = bwd_out[0][0].asnumpy()
        >>> weight0_grad = bwd_out[1][0].asnumpy()
        >>> weight1_grad = bwd_out[1][1].asnumpy()
        >>> print("x_grad", x_grad)
        x_grad [ 400. 1000. 1800.]
        >>> print("weight0_grad", weight0_grad)
        weight0_grad [2800. 4000. 5400.]
        >>> print("weight1_grad", weight1_grad)
        weight1_grad [ 700. 1600. 2700.]
    """
    @prim_attr_register
    def __init__(self, fn, infer_shape, infer_dtype, bprop_fn=None):
        self.add_prim_attr('side_effect_backprop', True)
        self.add_prim_attr('side_effect_mem', True)
        self.add_prim_attr('side_effect_io', True)
        self._infer_shape = infer_shape
        self._infer_dtype = infer_dtype

        self.add_prim_attr('__metamorphosis__', True)
        self.__morph_fn__ = fn
        self.__morph_bprop_fn__ = None
        if bprop_fn:
            self._check_fn_supported(fn)
            self.__morph_bprop_fn__ = bprop_fn

    def _check_fn_supported(self, fn):
        fn_sig = inspect.signature(fn)
        for param in fn_sig.parameters.values():
            if not (param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default is inspect.Parameter.empty):
                raise ValueError(f"When use `bprop` in Morph, Morph `fn` only support positional or keyword parameters "
                                 f"with default value is empty, but got param '{param.name}' "
                                 f"of kind '{param.kind.name}' with default value '{param.default}'.")

    def infer_shape(self, *args):
        return self._infer_shape(*args)

    def infer_dtype(self, *args):
        return self._infer_dtype(*args)

    def __call__(self, *args):
        raise RuntimeError("Morph is only supported in GRAPH_MODE.")


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
                       Default: ``""``, in this case, the system will automatically
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
        check_hook_fn(hook_fn)
        if cell_id != "":
            logger.warning(f"The args 'cell_id' of HookBackward will be removed in a future version. If the value of "
                           f"'cell_id' is set, the hook function will not work.")
        self.add_prim_attr("cell_id", cell_id)
        self.init_attrs["cell_id"] = cell_id
        self.cell_id = cell_id
        self.set_hook_fn(hook_fn, HookType.HookBackward)

    def infer_shape(self, *inputs_shape):
        if len(inputs_shape) == 1:
            return inputs_shape[0]
        return inputs_shape

    def infer_dtype(self, *inputs_type):
        for dtype in inputs_type:
            validator.check_subclass("input", dtype, [mstype.tensor_type], self.name)
        if len(inputs_type) == 1:
            return inputs_type[0]
        return inputs_type


class Print(Primitive):
    """
    Print the inputs to stdout.

    Refer to :func:`mindspore.ops.print_` for more detail.

    Inputs:
        - **input_x** (Union[Tensor, bool, int, float, str]) - The graph node to attach to.
          Supports multiple inputs which are separated by ','.

    Outputs:
        Tensor, has the same data type and shape as original `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn, ops
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
        [[1],
         [1]])
        Tensor(shape=[2, 2], dtype=Int32, value=
        [[1, 1],
         [1, 1]])
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Print."""
        if security.enable_security():
            raise ValueError(
                'The Print is not supported, please without `-s on` and recompile source.')
        self.add_prim_attr("side_effect_io", True)

    def __call__(self, *args):
        # Add for jit context.
        if jit_context() and jit_context().compiled:
            return
        for arg in args:
            if isinstance(arg, Parameter):
                print(Tensor_.__repr__(arg))
            elif isinstance(arg, (Tensor, Tensor_)):
                print(arg.__repr__())
            else:
                print(arg)
        # Add for jit context.
        if jit_context():
            jit_context().run_op(self, None, *args)


class Assert(PrimitiveWithInfer):
    """
    Asserts whether the given condition is True.
    If input condition is identified to be ``False``, print a list of the tensor in data.

    Args:
        summarize (int, optional): The number of entries to be printed in each tensor while the given condition is
            identified to be ``False`` . Default: ``3`` .

    Inputs:
        - **condition** (Union[Tensor[bool], bool]) - The condition to be identified.
        - **input_data** (Union[tuple[Tensor], list[Tensor]]) - The tensors to be printed out when the condition
          is ``False``.

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
          File "mindspore/graph/api.py", line 99, in wrapper
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
        validator.check_int(condition_len, 1, validator.LE, "condition's rank", self.name)
        if condition_len == 1:
            validator.check_equal_int(condition[0], 1, "condition[0]", self.name)
        return [1]

    def infer_dtype(self, condition, inputs):
        validator.check_scalar_or_tensor_types_same({"condition": condition}, [mstype.bool_], self.name)
        for dtype in inputs:
            validator.check_subclass("input", dtype, [mstype.tensor_type], self.name)
        return mstype.int32
