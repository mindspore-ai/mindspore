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
""" test_jit_config.py """
import os
import shutil
import subprocess
import pytest
import numpy as np
import mindspore as ms
from mindspore import context, jit, ops, nn, Tensor
from mindspore.ops.composite import GradOperation
from mindspore.ops.auto_generate.gen_ops_prim import select_ext_view_op, inplace_copy_op
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark


def get_jit_config_from_validate_ir(para, save_graphs_path):
    output_after = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" %
         (para, os.path.join(save_graphs_path, "[0-9]*_validate_[0-9]*.ir"))],
        shell=True)
    out_after = str(output_after, 'utf-8').strip()
    return out_after


def get_jit_config_from_jit_grad_ir(para, save_graphs_path):
    output_after = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" %
         (para, os.path.join(save_graphs_path, "call_graph_[0-9]*.ir"))],
        shell=True)
    out_after = str(output_after, 'utf-8').strip()
    return out_after


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_jit_grad_with_ms_backend():
    """
    Feature: Test setting ms_backend to jit in pynative grad.
    Description: Test setting ms_backend to jit in pynative grad.
    Expectation: success.
    """
    @jit(backend="ms_backend")
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    try:
        save_graphs_path = "./test_jit_grad_with_jit_config1"
        os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = save_graphs_path
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)
        a = Tensor([1, 2, 3])
        b = Tensor([1, 1, 1])
        ret = GradOperation()(func)(a, b)
        assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
        para = 'backend: ms_backend'
        out_front = get_jit_config_from_validate_ir(
            para, save_graphs_path)
        assert out_front == "1"
        out_back = get_jit_config_from_jit_grad_ir(para, save_graphs_path)
        assert out_back == "1"
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_multiple_jit_grad_with_ms_backend():
    """
    Feature: Test setting ms backend to multiple jit in pynative grad.
    Description: Test setting ms backend to multiple jit in pynative grad.
    Expectation: success.
    """
    @jit(backend="ms_backend")
    def inner_func1(x, y):
        return 2 * x[0] + y

    @jit(backend="ms_backend")
    def inner_func2(x, y):
        return 2 * x[0] - y

    @jit(backend="ms_backend")
    def inner_func3(x, y):
        return 2 * x[0] * y

    def func(x, y):
        x = x * 3
        return inner_func1((x,), y) + inner_func2((x,), y) + inner_func3((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    try:
        save_graphs_path = "./test_jit_grad_with_jit_config2"
        os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = save_graphs_path
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)
        a = Tensor([1, 2, 3])
        b = Tensor([1, 1, 1])
        ret = GradOperation()(func)(a, b)
        assert np.all(ret.asnumpy() == np.array([18, 18, 18]))
        para = 'backend: ms_backend'
        out_front1 = get_jit_config_from_validate_ir(
            para, save_graphs_path)
        assert out_front1 == "3"
        out_back1 = get_jit_config_from_jit_grad_ir(para, save_graphs_path)
        assert out_back1 == "3"
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_multiple_jit_grad_with_different_backend():
    """
    Feature: Test setting different backend to multiple jit in pynative grad.
    Description: Test setting different backend to multiple jit in pynative grad.
    Expectation: success.
    """
    @jit(backend="ms_backend")
    def inner_func1(x, y):
        return 2 * x[0] + y

    @jit(backend="GE")
    def inner_func2(x, y):
        return 2 * x[0] - y

    @jit(backend="ms_backend")
    def inner_func3(x, y):
        return 2 * x[0] * y

    def func(x, y):
        x = x * 3
        return inner_func1((x,), y) + inner_func2((x,), y) + inner_func3((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    try:
        save_graphs_path = "./test_jit_grad_with_jit_config3"
        os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = save_graphs_path
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)
        a = Tensor([1, 2, 3])
        b = Tensor([1, 1, 1])
        ret = GradOperation()(func)(a, b)
        assert np.all(ret.asnumpy() == np.array([18, 18, 18]))
        para = 'backend: ms_backend'
        out_front1 = get_jit_config_from_validate_ir(
            para, save_graphs_path)
        assert out_front1 == "2"
        out_back1 = get_jit_config_from_jit_grad_ir(para, save_graphs_path)
        assert out_back1 == "2"
        para = 'backend: GE'
        out_front2 = get_jit_config_from_validate_ir(
            para, save_graphs_path)
        assert out_front2 == "1"
        out_back2 = get_jit_config_from_jit_grad_ir(para, save_graphs_path)
        assert out_back2 == "1"
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        if os.path.exists(save_graphs_path):
            shutil.rmtree(save_graphs_path)


@pytest.mark.skip(reason="jit grad cannot set to ge mode right now.")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_set_ge_raise_error():
    """
    Feature: Set backend to ge, raise error if success.
    Description: Set backend to ge, raise error if success.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x = ms.Tensor([[0, 1], [2, 3]], dtype=ms.float32)
    net = Net()
    net.construct = ms.jit(net.construct, backend="GE")
    with pytest.raises(RuntimeError) as err:
        grad(net)(x)
    assert "ge_backend" in str(err.value)
