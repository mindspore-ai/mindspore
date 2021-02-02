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
"""
Graph kernels. They are composites of basic primitives and can be compiled into
a fused kernel automatically when context.set_context(enable_graph_kernel=True).
"""
from ...ops import operations as P
from ...ops.primitive import PrimitiveWithInfer, prim_attr_register
from ...ops.composite import multitype_ops as C
from ..cell import GraphKernel


class InplaceAssign(PrimitiveWithInfer):
    """
    Inplace assign `Parameter` with a value.

    This primitive can only be used in graph kernel.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Tensor) - The value to be assigned.
        - **depend** (Tensor) - The dependent tensor to keep this op connected in graph.

    Outputs:
        Tensor, has the same type as original `variable`.

    Examples:
        >>> class MyClass(GraphKernel):
        ...     def __init__(self):
        ...         super(MyClass, self).__init__()
        ...         self.mul = P.Mul()
        ...         self.fake_output_assign = InplaceAssign()
        ...         self.fake_output_assign.add_prim_attr("fake_output", True)
        ...
        ...     def construct(self, i0, i1):
        ...         mul_res = self.mul(i0, i1)
        ...         # mul_res is a fake output and parameter i0 will be updated.
        ...         mul_res = self.fake_output_assign(i0, mul_res, mul_res)
        ...         return mul_res
    """

    @prim_attr_register
    def __init__(self):
        super(InplaceAssign, self).__init__("InplaceAssign")
        self.init_prim_io_names(inputs=['x', 'y', 'z'], outputs=['output'])

    def infer_shape(self, x, y, z):
        return z

    def infer_dtype(self, x, y, z):
        return z

    def get_bprop(self):
        def bprop(x, y, z, out, dout):
            return (x, C.zeros_like(y), dout)

        return bprop


class LambUpdateWithLR(GraphKernel):
    r"""
    Part of Lamb optimizer.

    .. math::
        s_1 = select(i_1 \gt y_g, select(i_0 \gt y_g, \frac{i_1}{i_2}, se), se)
        i_5 = i_5 - max(min(s_1, y_m), y_g) \times i_3 \times i_4

    Inputs:
        - **input0** (Tensor) - The first tensor to be computed.
        - **input1** (Tensor) - The second tensor to be computed.
        - **input2** (Tensor) - The third tensor to be computed.
        - **input3** (Tensor) - The fourth tensor to be computed.
        - **input4** (Tensor) - The fifth tensor to be computed.
        - **input5** (Tensor) - The sixth tensor to be computed. It will be updated by result.
        - **greater_y** (Tensor) - The seventh tensor to be computed.
        - **select_e** (Tensor) - The eighth tensor to be computed.
        - **minimum_y** (Tensor) - The ninth tensor to be computed.

    Outputs:
        A fake output tensor.

    Examples:
        >>> import numpy as np
        >>> import mindspore.context as context
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.common.tensor import Tensor
        >>> from mindspore.common.parameter import Parameter
        >>> from mindspore.nn.cell import Cell
        >>> class Net(Cell):
        ...     def __init__(self, i5):
        ...         super(Net, self).__init__()
        ...         self.i5 = Parameter(i5, name='i5')
        ...         self.lamb_update = LambUpdateWithLR()
        ...
        ...     def construct(self, i0, i1, i2, i3, i4, i6, i7, i8):
        ...         return self.lamb_update(i0, i1, i2, i3, i4, self.i5, i6, i7, i8)
        >>> shape = [1, 16]
        >>> oshape = [1]
        >>> i0 = Tensor(np.random.normal(0, 1, oshape).astype(np.float32))
        >>> i1 = Tensor(np.random.normal(0, 1, oshape).astype(np.float32))
        >>> i2 = Tensor(np.random.normal(0, 1, oshape).astype(np.float32))
        >>> i3 = Tensor(np.random.normal(0, 1, oshape).astype(np.float32))
        >>> i4 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i5 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i6 = Tensor(np.random.normal(0, 1, oshape).astype(np.float32))
        >>> i7 = Tensor(np.random.normal(0, 1, oshape).astype(np.float32))
        >>> i8 = Tensor(np.random.normal(0, 1, oshape).astype(np.float32))
        >>> context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
        >>> net = Net(i5)
        >>> _ = net(i0, i1, i2, i3, i4, i6, i7, i8)
        >>> output = (net.i5)
    """

    def __init__(self):
        super(LambUpdateWithLR, self).__init__()
        self.greater = P.Greater()
        self.select = P.Select()
        self.div = P.RealDiv()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.fake_output_assign = InplaceAssign()
        self.fake_output_assign.add_prim_attr("fake_output", True)

    def construct(self, input0, input1, input2, input3, input4, input5, greater_y, select_e, minimum_y):
        greater0 = self.greater(input0, greater_y)
        greater1 = self.greater(input1, greater_y)
        real_div0 = self.div(input1, input2)
        select0 = self.select(greater0, real_div0, select_e)
        select1 = self.select(greater1, select0, select_e)
        min0 = self.min(select1, minimum_y)
        max0 = self.max(min0, greater_y)
        mul0 = self.mul(max0, input3)
        mul1 = self.mul(mul0, input4)
        sub0 = self.sub(input5, mul1)
        sub0 = self.fake_output_assign(input5, sub0, sub0)
        return sub0


class LambNextMV(GraphKernel):
    r"""
    Part of Lamb optimizer.

    .. math::
        rd_0 = \frac{i_8 \times i_5 + i_9 \times i_4}{i6}
        rd_1 = \frac{x_0 \times i_2 + x_1 \times i_1}{i3}
        y_2 = \frac{rd_0}{\sqrt{rd_1 + x3}} + x_2 \times i_7
        y_3 = \frac{rd_0}{\sqrt{rd_1} + x3}
        i5 = i_8 \times i_5 + i_9 \times i_4
        i2 = x_0 \times i_2 + x_1 \times i_1

    Inputs:
        - **inputs1** (Tensor) - The first input tensor to be computed.
        - **inputs2** (Tensor) - The second input tensor to be computed. It will be updated by result.
        - **inputs3** (Tensor) - The third input tensor to be computed.
        - **inputs4** (Tensor) - The fourth input tensor to be computed.
        - **inputs5** (Tensor) - The fifth input tensor to be computed. It will be updated by result.
        - **inputs6** (Tensor) - The sixth input tensor to be computed.
        - **inputs7** (Tensor) - The seventh input tensor to be computed.
        - **inputs8** (Tensor) - The eighth input tensor to be computed.
        - **inputs9** (Tensor) - The ninth input tensor to be computed.
        - **inputsx0** (Tensor) - The tenth input tensor to be computed.
        - **inputsx1** (Tensor) - The eleventh input tensor to be computed.
        - **inputsx2** (Tensor) - The twelfth input tensor to be computed.
        - **inputsx3** (Tensor) - The thirteenth input tensor to be computed.

    Outputs:
        Tuple of 2 Tensors.

        - **add3** (Tensor) - the shape is the same as the one after broadcasting, and the data type is
                              the one with higher precision or higher digits among the inputs.
        - **realdiv4** (Tensor) - the shape is the same as the one after broadcasting, and the data type is
                                  the one with higher precision or higher digits among the inputs.

    Examples:
        >>> import numpy as np
        >>> import mindspore.context as context
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.common.tensor import Tensor
        >>> from mindspore.common.parameter import Parameter
        >>> from mindspore.nn.cell import Cell
        >>> class Net(Cell):
        ...     def __init__(self, i1, i4):
        ...         super(Net, self).__init__()
        ...         self.i1 = Parameter(i1, name='i1')
        ...         self.i4 = Parameter(i4, name='i4')
        ...         self.lamb_next = LambNextMV()
        ...
        ...     def construct(self, i0, i2, i3, i5, i6, i7, i8, i9, i10, i11, i12):
        ...         i0_ = i0 + i2
        ...         return self.lamb_next(i0_, self.i1, i2, i3, self.i4, i5, i6, i7, i8, i9, i10, i11, i12)
        >>> shape = [1, 16]
        >>> i0 = Tensor(np.abs(np.random.normal(0, 1, shape)).astype(np.float32))
        >>> i1 = Tensor(np.abs(np.random.normal(0, 1, shape)).astype(np.float32))
        >>> i2 = Tensor(np.abs(np.random.normal(0, 1, shape)).astype(np.float32))
        >>> i3 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i4 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i5 = Tensor(np.abs(np.random.normal(0, 1, shape)).astype(np.float32))
        >>> i6 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i7 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i8 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i9 = Tensor(np.abs(np.random.normal(0, 1, shape)).astype(np.float32))
        >>> i10 = Tensor(np.abs(np.random.normal(0, 1, shape)).astype(np.float32))
        >>> i11 = Tensor(np.random.normal(0, 1, shape).astype(np.float32))
        >>> i12 = Tensor(np.ones(shape).astype(np.float32) * 1e-6)
        >>> context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
        >>> net = Net(i1, i4)
        >>> (o0, o1) = net(i0, i2, i3, i5, i6, i7, i8, i9, i10, i11, i12)
        >>> output = (o0, net.i4, net.i1, o1)
    """

    def __init__(self):
        super(LambNextMV, self).__init__()
        self.mul = P.Mul()
        self.add = P.Add()
        self.div = P.RealDiv()
        self.sqrt = P.Sqrt()
        self.rsqrt = P.Rsqrt()
        self.fake_output_assign_1 = InplaceAssign()
        self.fake_output_assign_1.add_prim_attr("fake_output", False)
        self.fake_output_assign_2 = InplaceAssign()
        self.fake_output_assign_2.add_prim_attr("fake_output", False)

    def construct(self, input1, input2, input3, input4, input5, input6, input7,
                  input8, input9, inputx0, inputx1, inputx2, inputx3):
        mul3 = self.mul(inputx1, input1)
        mul2 = self.mul(inputx0, input2)
        add1 = self.add(mul2, mul3)
        realdiv1 = self.div(add1, input3)
        add2 = self.add(realdiv1, inputx3)
        sqrt0 = self.rsqrt(add2)
        sqrt1 = self.sqrt(realdiv1)
        add4 = self.add(sqrt1, inputx3)
        mul1 = self.mul(input9, input4)
        mul0 = self.mul(input8, input5)
        add0 = self.add(mul0, mul1)
        realdiv0 = self.div(add0, input6)
        realdiv2 = self.mul(realdiv0, sqrt0)
        realdiv4 = self.div(realdiv0, add4)
        mul4 = self.mul(inputx2, input7)
        add3 = self.add(realdiv2, mul4)

        add3 = self.fake_output_assign_1(input5, add0, add3)
        add3 = self.fake_output_assign_2(input2, add1, add3)

        return add3, realdiv4
