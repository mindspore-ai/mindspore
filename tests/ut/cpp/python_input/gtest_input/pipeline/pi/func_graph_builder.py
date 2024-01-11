# Copyright 2024 Huawei Technologies Co., Ltd
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

from mindspore.ops import Primitive
from mindspore.ops.operations import _scalar_ops

make_tuple = Primitive("MakeTuple")


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


# pylint: disable=unused-variable
def test_add_inputs_and_outputs(tag):
    """
    Feature: Build graph in pi_jit.
    Description: Use the func_graph_builder api to add inputs and add outputs.
    Expectation: The expected graph is constructed.
    """
    fns = FnDict()

    @fns
    def graph(x, y):
        return y

    return fns[tag]


def test_add_node(tag):
    """
    Feature: Build graph in pi_jit.
    Description: Use the func_graph_builder api to add cnode.
    Expectation: The expected graph is constructed.
    """
    fns = FnDict()
    scalar_add = _scalar_ops.ScalarAdd()

    @fns
    def graph_single_output(x, y):
        return scalar_add(x, y)

    @fns
    def graph_multi_output(x, y):
        out = scalar_add(x, y)
        return make_tuple(out, out)

    return fns[tag]


def test_add_node_with_constant(tag):
    """
    Feature: Build graph in pi_jit.
    Description: Use the func_graph_builder api to add cnode.
    Expectation: The expected graph is constructed.
    """
    fns = FnDict()
    scalar_add = _scalar_ops.ScalarAdd()

    @fns
    def graph(x):
        return scalar_add(x, 2)

    return fns[tag]


def test_add_binary_node(tag):
    """
    Feature: Build graph in pi_jit.
    Description: Use the func_graph_builder api to add cnode.
    Expectation: The expected graph is constructed.
    """
    fns = FnDict()

    @fns
    def graph(x, y):
        return x + y

    return fns[tag]


def test_remove_output(tag):
    """
    Feature: Build graph in pi_jit.
    Description: Use the func_graph_builder api to remove an output.
    Expectation: The expected graph is constructed.
    """
    fns = FnDict()
    scalar_add = _scalar_ops.ScalarAdd()

    @fns
    def graph(x, y, z):
        return scalar_add(y, z)

    return fns[tag]


def test_add_fg_call_node(tag):
    """
    Feature: Build graph in pi_jit.
    Description: Use the func_graph_builder api to add func_graph called node.
    Expectation: The expected graph is constructed.
    """
    fns = FnDict()
    scalar_add = _scalar_ops.ScalarAdd()

    def graph1(x, y):
        return scalar_add(x, y)

    def graph2(x, y):
        return make_tuple(scalar_add(x, y), scalar_add(x, y))

    @fns
    def graph_single_output(x, y):
        return graph1(x, y)

    @fns
    def graph_multi_output(x, y):
        return graph2(x, y)

    return fns[tag]
