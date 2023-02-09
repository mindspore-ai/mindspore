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
# ============================================================================
""" cse_test """
import mindspore as ms
from mindspore import ops
from mindspore import Tensor


# pylint: disable=W0612


class TestCSEFnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        try:
            return self.fn_dict[name]
        except KeyError:
            return None


def test_has_hidden_side_effect(tag):
    """
    Feature: CSE.
    Description: Create graphs with hiden_side_effect attr.
    Expectation: Func graphs are successfully created.
    """

    fns = TestCSEFnDict()
    x = Tensor([1], ms.int32)
    pow_ops = ops.Pow()

    def normal_node_graph(x):
        return ops.add(x, x)

    def hidden_side_effect_node_graph(input_shape):
        return ops.StandardNormal(0, 1)(input_shape)

    def hidden_side_effect_node_parent_graph(input_shape):
        return hidden_side_effect_node_graph(input_shape)

    @fns
    def root_graph_normal_call():
        pow_out = pow_ops(x, x)
        out1 = normal_node_graph(pow_out)
        out2 = normal_node_graph(pow_out)
        return ops.make_tuple(out1, out2)

    print(root_graph_normal_call)

    @fns
    def root_graph_hidden_side_effect_call():
        shape = (1, 2, 3)
        out1 = hidden_side_effect_node_parent_graph(shape)
        out2 = hidden_side_effect_node_parent_graph(shape)
        return ops.make_tuple(out1, out2)

    return fns[tag]
