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
# ============================================================================

import functools
import os
import shutil
import subprocess
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops.auto_generate.gen_ops_prim import select_ext_view_op, inplace_copy_op
from mindspore.ops.functional import grad

# IR file pattern
virtual_view_grad_insert_pattern = '*_virtual_view_grad_insert_*.ir'
virtual_view_insert_pattern = '*_virtual_view_insert_*.ir'
remove_redundant_virtual_ops_pattern = '*_remove_redundant_virtual_ops_*.ir'


def save_graph_setting(graph_save_path):
    os.environ['MS_DEV_SAVE_GRAPHS'] = '3'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = graph_save_path
    if os.path.exists(graph_save_path):
        shutil.rmtree(graph_save_path)


def remove_graph_path(graph_save_path):
    os.unsetenv('MS_DEV_SAVE_GRAPHS')
    os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
    if os.path.exists(graph_save_path):
        shutil.rmtree(graph_save_path)


def check_prim_number_valid(prim_name, ir_pattern, size, graph_save_path):
    file_path = os.path.join(graph_save_path, ir_pattern)
    para = prim_name
    output = subprocess.check_output(["grep -r '%s' %s | wc -l" % (para, file_path)], shell=True)
    out = str(output, 'utf-8').strip()
    assert out == str(size)


@ms.jit
def grad_under_jit(net, arg):
    return grad(net)(arg)


def save_graphs_for_case(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        graph_save_path = func.__name__
        save_graph_setting(graph_save_path)
        func(*args, **kwargs)
        remove_graph_path(graph_save_path)
    return wrapper


@save_graphs_for_case
def test_virtualviewgrad_number_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Verify that the number of VirtualViewOps in IR.
    Expectation: Match the expected count.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            y_viewed2 = select_ext_view_op(y_viewed, 0, 0)
            y_viewed3 = select_ext_view_op(y_viewed2, 0, 0)
            inplace_copy_op(y_viewed3, ms.Tensor(-1, dtype=ms.float32))
            return y

    x = ms.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=ms.float32)
    net = Net()
    grad_under_jit(net, x)
    check_prim_number_valid('_VirtualViewGrad(%', remove_redundant_virtual_ops_pattern,
                            3, "test_virtualviewgrad_number_case1")


@save_graphs_for_case
def test_virtualviewgrad_number_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Verify that the number of VirtualViewOps in IR.
    Expectation: Match the expected count.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            y_viewed2 = select_ext_view_op(y_viewed, 0, 0)
            y_viewed3 = select_ext_view_op(y_viewed2, 0, 0)
            inplace_copy_op(y_viewed3, ms.Tensor(-1, dtype=ms.float32))
            return y_viewed

    x = ms.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=ms.float32)
    net = Net()
    grad_under_jit(net, x)
    dir_path = "test_virtualviewgrad_number_case2"
    check_prim_number_valid('_VirtualViewGrad(%', virtual_view_grad_insert_pattern, 3, dir_path)
    check_prim_number_valid('_VirtualViewGrad(%', remove_redundant_virtual_ops_pattern, 2, dir_path)


@save_graphs_for_case
def test_virtualview_number_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Verify that the number of VirtualViewOps in IR.
    Expectation: Match the expected count.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed1 = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y, ms.Tensor(-1, dtype=ms.float32))
            y_viewed2 = select_ext_view_op(y, 0, 1)
            inplace_copy_op(y_viewed1, ms.Tensor(-1, dtype=ms.float32))
            inplace_copy_op(y_viewed2, ms.Tensor(-1, dtype=ms.float32))
            return y

    x = ms.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=ms.float32)
    net = Net()
    grad_under_jit(net, x)
    dir_path = "test_virtualview_number_case1"
    check_prim_number_valid('{is_virtual_view_op: Bool(1)}', virtual_view_insert_pattern, 2, dir_path)
    check_prim_number_valid('{is_virtual_view_op: Bool(1)}', remove_redundant_virtual_ops_pattern, 2, dir_path)
    check_prim_number_valid('_VirtualViewGrad(%', remove_redundant_virtual_ops_pattern, 2, dir_path)


@save_graphs_for_case
def test_virtualview_number_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Verify that the number of VirtualViewOps in IR.
    Expectation: Match the expected count.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed1 = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y, ms.Tensor(-1, dtype=ms.float32))
            inplace_copy_op(y_viewed1, ms.Tensor(-1, dtype=ms.float32))
            y_viewed2 = select_ext_view_op(y, 0, 1)
            inplace_copy_op(y_viewed1, ms.Tensor(-5, dtype=ms.float32))
            inplace_copy_op(y_viewed2, ms.Tensor(-1, dtype=ms.float32))
            return y_viewed1

    x = ms.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=ms.float32)
    net = Net()
    grad_under_jit(net, x)
    dir_path = "test_virtualview_number_case2"
    check_prim_number_valid('{is_virtual_view_op: Bool(1)}', virtual_view_insert_pattern, 4, dir_path)
    check_prim_number_valid('{is_virtual_view_op: Bool(1)}', remove_redundant_virtual_ops_pattern, 4, dir_path)
    check_prim_number_valid('_VirtualViewGrad(%', remove_redundant_virtual_ops_pattern, 3, dir_path)


@save_graphs_for_case
def test_remove_virtualviewops_case():
    """
    Feature: Support tensor inplace view gradient.
    Description: Verify that the number of VirtualViewOps in IR.
    Expectation: Match the expected count.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed1 = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y_viewed1, ms.Tensor(-1, dtype=ms.float32))
            inplace_copy_op(y, ms.Tensor(-1, dtype=ms.float32))
            y_viewed2 = select_ext_view_op(y_viewed1, 0, 1)
            inplace_copy_op(y_viewed2, ms.Tensor(-1, dtype=ms.float32))
            return y_viewed1

    x = ms.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=ms.float32)
    net = Net()
    grad_under_jit(net, x)
    dir_path = "test_remove_virtualviewops_case"
    check_prim_number_valid('_VirtualViewGrad(%', virtual_view_grad_insert_pattern, 3, dir_path)
    check_prim_number_valid('{is_virtual_view_op: Bool(1)}', virtual_view_insert_pattern, 2, dir_path)
    check_prim_number_valid('{is_virtual_view_op: Bool(1)}', remove_redundant_virtual_ops_pattern, 1, dir_path)
    check_prim_number_valid('_VirtualViewGrad(%', remove_redundant_virtual_ops_pattern, 2, dir_path)
