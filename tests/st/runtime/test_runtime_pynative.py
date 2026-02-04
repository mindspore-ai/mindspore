# Copyright 2025 Huawei Technologies Co., Ltd
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

import numpy as np
from mindspore import context, nn, Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.train.model import Model
from tests.mark_utils import arg_mark

class AssignSubNet(nn.Cell):
    def __init__(self, ref, ref_reduce):
        super().__init__()
        self.unique = P.Unique()
        self.assign_sub = P.AssignSub()
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.ref = Parameter(ref)
        self.cast = P.Cast()
        self.ref_reduce = Parameter(ref_reduce)

    def construct(self, x, indices):
        x = self.assign_sub(self.ref, x)
        unique_indices, _ = self.unique(indices)
        dtype = x.dtype
        x = self.cast(x, mstype.float64)
        x = self.reducesum(x, unique_indices)
        x = self.cast(x, dtype)
        out = self.assign_sub(self.ref_reduce, x)
        return x, out


class AssignSubDynamicShape():
    def __init__(self, input_ref_np, input_value_np, indices_np):
        self.input_ref_np = input_ref_np
        self.input_value_np = input_value_np
        self.indices_np = indices_np
        self.unique = P.Unique()
        self.ref_reduce_np = None
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.cast = P.Cast()

    def impl(self):
        indices_ms = Tensor(self.indices_np)
        unique_indices, _ = self.unique(indices_ms)
        ref_reduce = Tensor(self.input_ref_np)
        dtype = ref_reduce.dtype
        ref_reduce = self.cast(ref_reduce, mstype.float32)
        ref_reduce = self.reducesum(ref_reduce, unique_indices.asnumpy().tolist())
        ref_reduce = self.cast(ref_reduce, dtype)
        self.ref_reduce_np = ref_reduce.asnumpy()
        net = AssignSubNet(Tensor(self.input_ref_np), ref_reduce)
        input_ms = Tensor(self.input_value_np)
        input_dyn = Tensor(shape=[None for _ in input_ms.shape], dtype=input_ms.dtype)
        net.set_inputs(input_dyn, indices_ms)
        out_ms = net(input_ms, indices_ms)
        return out_ms


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_single_op_graph():
    """
    Feature: Runtime special output.
    Description: Test the output is the depend with value node, that the value can't be converted the tensor.
    Expectation: Not throw exception.
    """
    np.random.seed(0)
    input_ref_np = np.random.randn(128, 128, 32).astype(np.float16)
    input_value_np = np.random.randn(128, 128, 32).astype(np.float16)
    indices_np = np.random.randint(0, 2, size=2)
    context.set_context(device_target="CPU")
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = AssignSubDynamicShape(input_ref_np, input_value_np, indices_np)
    result1 = net1.impl()

    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    net2 = AssignSubDynamicShape(input_ref_np, input_value_np, indices_np)
    result2 = net2.impl()
    assert np.allclose(result1[0].asnumpy(), result2[0].asnumpy())



@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=820.00)
def test_format_trans():
    """
    Feature: test different format between pynative and graph.
    Description: pynative to graph mode.
    Expectation: Not throw exception.
    """
    class NetConv2d(nn.Cell):
        def __init__(self, in_channel, out_channel, kernel_size, stride_size, kernel_me):
            super().__init__()
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride_size, padding=0, has_bias=False,
                                  weight_init=Tensor(kernel_me), pad_mode="valid")

        def construct(self, x):
            x = self.conv(x)
            return x

    class NetBatchNorm2d(nn.Cell):
        def __init__(self, out_channel, kernel_bn, bias_bn, eps):
            super().__init__()
            self.bn = nn.BatchNorm2d(num_features=out_channel, eps=eps, beta_init=Tensor(bias_bn),
                                     gamma_init=Tensor(kernel_bn))

        def construct(self, x):
            x = self.bn(x)
            return x

    input_shape = (32, 3, 224, 224)
    output_channel = 64
    kernel_size = 6
    stride_size = 1
    n, in_channel, h, w = input_shape
    input_np = np.random.randn(n, h, w, in_channel).astype(np.float32)
    eps = 1e-3
    kernel = np.random.randn(kernel_size, kernel_size, in_channel, output_channel).astype(np.float32)
    kernel_bn = np.random.randn(output_channel).astype(np.float32)
    bias_bn = np.random.randn(output_channel).astype(np.float32)
    input_me = input_np.transpose(0, 3, 1, 2)
    kernel_me = kernel.transpose(3, 2, 0, 1)
    net_a = NetConv2d(in_channel, output_channel, kernel_size, stride_size, kernel_me)
    output_conv2d = net_a(Tensor(input_me))
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
    net_b = NetBatchNorm2d(output_channel, kernel_bn, bias_bn, eps)
    model_graph = Model(net_b)
    out_graph = model_graph.predict(Tensor(output_conv2d))
    context.set_context(mode=context.PYNATIVE_MODE)
    net_b_1 = NetBatchNorm2d(output_channel, kernel_bn, bias_bn, eps)
    model_pynative = Model(net_b_1)
    out_pynative = model_pynative.predict(Tensor(output_conv2d))
    assert np.allclose(out_graph.asnumpy(), out_pynative.asnumpy())
