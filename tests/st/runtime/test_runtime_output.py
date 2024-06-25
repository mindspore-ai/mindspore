# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import context, nn, Tensor, jit, ops
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


class NetValueNodeWithDepend(nn.Cell):
    def construct(self, input_x):
        print(input_x[1][1])
        output = input_x[1]
        output[1] = 0
        return output


class NetSubGraphOutputWithLoad(nn.Cell):
    def __init__(self):
        super(NetSubGraphOutputWithLoad, self).__init__()
        self.bias = Parameter(Tensor(np.ones([10])).astype(np.float32), name="bias1")
        self.biass_add1 = P.BiasAdd()
        self.biass_add2 = P.BiasAdd()

    def construct(self, input_x):
        output = self.biass_add1(input_x, self.bias)
        output = self.biass_add2(output, self.bias)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_value_node_with_depend():
    """
    Feature: Runtime special output.
    Description: Test the output is the depend with value node, that the value can't be converted the tensor.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = [[1, 2, 3, 4], [5, 6, 7, 8]]
    net = NetValueNodeWithDepend()
    output = net(x)
    assert output == [5, 0, 7, 8]


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_subgraph_output_with_load():
    """
    Feature: Runtime special subgraph output.
    Description: Test the subgraph output is the load node.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.ones([32, 10])).astype(np.float32)
    net1 = NetSubGraphOutputWithLoad()
    output1 = net1(x)
    net2 = NetSubGraphOutputWithLoad()
    net2.biass_add2.add_prim_attr("primitive_target", "CPU")
    output2 = net2(x)
    assert (output1 == output2).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_runtime_heter():
    """
    Feature: Runtime heter.
    Description: Test multi graph share same parameter.
    Expectation: Not throw exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    mul = P.Mul().add_prim_attr("primitive_target", "CPU")
    add = P.Add()

    @jit
    def foo(a, b):
        c = add(a, b)
        d = mul(a, c)
        e = add(a, d)
        f = mul(a, e)
        return f
    ret = foo(Tensor(1), Tensor(2))
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@test_utils.run_test_with_On
def test_runtime_fallback_heter():
    """
    Feature: Runtime heter.
    Description: Test any type kernel actor link to copy actor.
    Expectation: Not throw exception.
    """

    @jit
    def foo():
        d = {'a': 1, 'b': 2, 'c': 3, 'A': 4, 'B': 5, 'D': 6}
        res = {i.lower(): d.get(i.lower(), 0) + d.get(i.upper(), 0) for i in d}
        return res

    ret = foo()
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_runtime_value_tuple_output():
    """
    Feature: Runtime tuple output to make tuple.
    Description: value tuple used more than once.
    Expectation: Not throw exception.
    """

    @jit
    def foo(a):
        b = (2, 3)
        return (ops.reshape(a, b), b)

    b = Tensor([1, 2, 3, 4, 5, 6])
    ret = foo(b)
    assert ret
