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
from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import Primitive, prim_attr_register, PrimitiveWithInfer


class ScalarSummary(Primitive):
    """
    Output scalar to protocol buffer through scalar summary operator.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of scalar.

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

    def __call__(self, *args, **kwargs):
        pass


class ImageSummary(Primitive):
    """
    Output image tensor to protocol buffer through image summary operator.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of image.

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

    def __call__(self, *args, **kwargs):
        pass


class TensorSummary(Primitive):
    """
    Output tensor to protocol buffer through tensor summary operator.

    Inputs:
        - **name** (str) - The name of the input variable.
        - **value** (Tensor) - The value of tensor.

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

    def __call__(self, *args, **kwargs):
        pass


class HistogramSummary(Primitive):
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


class InsertGradientOf(PrimitiveWithInfer):
    """
    Attach callback to graph node that will be invoked on the node's gradient.

    Args:
        f (Function): MindSpore's Function. Callback function.

    Inputs:
        - **input_x** (Tensor) - The graph node to attach to.

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


class Print(PrimitiveWithInfer):
    """
    Output tensor or string to stdout.

    Note:
        The print operation cannot support the following cases currently.

        1. The type of tensor is float64 or bool.

        2. The data of tensor is a scalar type.

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
