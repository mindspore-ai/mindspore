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

import os
import pytest
import shutil
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, Symbol, enable_dynamic
from mindspore.ops import functional as F
from mindspore.graph.api import jit


def check_ir_symbolic_shape(dir_path, target_str, expect_num):
    """Check ir if exist symbolic shape"""
    ir_validate_num = 0
    found_target = False
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"'{dir_path}' is not a valid directory.")

    for filename in os.listdir(dir_path):
        if 'validate' in filename:
            ir_validate_num = ir_validate_num + 1
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                content = f.read()
                if target_str in content:
                    found_target = True

    assert ir_validate_num == expect_num, f"Expect ir number is '{expect_num}', but got '{ir_validate_num}'!"
    assert found_target, f"Dynamic shape without symbolic info!"
    shutil.rmtree(dir_path)


def test_enable_dynamic_symbolic_shape_inputs():
    """
    Feature: Dynamic shape with symbolic info
    Description: Test symbolic_shape by set enable_dynamic
    Expectation: success
    """
    s1 = Symbol(max=6, divisor=3) # the value can be 3, 6
    s2 = Symbol(unique=True)
    x_dyn = Tensor(shape=[1, s1, s2], dtype=ms.float32)
    y_dyn = Tensor(shape=[1, s2, s1], dtype=ms.float32)

    @enable_dynamic(x=x_dyn, y=y_dyn)
    @jit
    def add_func(x, y):
        return F.tensor_add(x, y)

    with pytest.raises(ValueError) as e1:
        x = Tensor(np.ones((1, 12, 12), np.float32))
        add_func(x, x)  # s1 > max
    assert "The 2th shape value of 1th actual input args must be" in str(e1.value)

    with pytest.raises(ValueError) as e2:
        x = Tensor(np.ones((1, 3, 6), np.float32))
        y = Tensor(np.ones((1, 1, 6), np.float32))
        add_func(x, y)  # s2 is unique, but y.shape[1] != x.shape[2]
    assert "The 2th shape value of 2th actual input args is a unique symbol" in str(e2.value)

    with pytest.raises(ValueError) as e3:
        x = Tensor(np.ones((1, 5, 5), np.float32))
        add_func(x, x)  # s1.divisor = 3, but x.shape[1] == 5
    assert "The 2th shape value of 1th actual input args must be match the 'divisor'" in str(e3.value)

    dir_path = 'ir_enable_dynamic_symbolic_shape_inputs'
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = dir_path
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' has been deleted.")

    x = Tensor(np.ones((1, 3, 3), np.float32))
    add_func(x, x)
    check_ir_symbolic_shape(dir_path, '-> (S', 1)
    os.unsetenv('MS_DEV_SAVE_GRAPHS')
    os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')


def test_enable_dynamic_symbolic_shape_grad():
    """
    Feature: Dynamic shape with symbolic info
    Description: Test symbolic_shape by set enable_dynamic
    Expectation: success
    """
    dir_path = 'ir_enable_dynamic_symbolic_shape_grad'
    target_str = '-> (S'
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = dir_path
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' has been deleted.")

    s1 = Symbol(max=10, unique=True)
    s2 = Symbol(min=2, unique=True)
    x_dyn = Tensor(shape=[1, s1, s1], dtype=ms.float32)
    y_dyn = Tensor(shape=[2, s2, s2], dtype=ms.float32)

    def add_func(x, y):
        return F.tensor_add(x, y)

    @enable_dynamic(x=x_dyn, y=y_dyn)
    @jit
    def grad_add_func(foo, x, y):
        return ops.grad(foo)(x, y) # pylint: disable=not-callable

    x = Tensor(np.ones((1, 8, 8), np.float32))
    y = Tensor(np.ones((2, 8, 8), np.float32))
    grad_add_func(add_func, x, y)
    check_ir_symbolic_shape(dir_path, target_str, 1)
    os.unsetenv('MS_DEV_SAVE_GRAPHS')
    os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')


def test_enable_auto_dynamic_symbolic_shape():
    """
    Feature: Dynamic shape with symbolic info
    Description: Test symbolic_shape by set enable_dynamic and dynamic
    Expectation: success
    """
    dir_path = 'ir_enable_auto_dynamic_symbolic_shape'
    target_str = '<[1,inf]>'
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = dir_path
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' has been deleted.")

    s1 = Symbol(min=2, unique=True)
    x_dyn = Tensor(shape=[2, s1], dtype=ms.float32)

    @enable_dynamic(x=x_dyn)
    @jit(dynamic=1)
    def func(x, y):
        return x + 1, y + 1


    x1 = Tensor(np.random.randn(2, 2), ms.float32)
    x2 = Tensor(np.random.randn(2, 3), ms.float32)
    x3 = Tensor(np.random.randn(2, 4), ms.float32)
    x4 = Tensor(np.random.randn(2, 5), ms.float32)

    y1 = Tensor(np.random.randn(1, 3), ms.float32)
    y2 = Tensor(np.random.randn(2, 3), ms.float32)
    y3 = Tensor(np.random.randn(3, 3), ms.float32)
    y4 = Tensor(np.random.randn(4, 3), ms.float32)

    func(x1, y1)
    func(x2, y2)
    func(x3, y3)
    func(x4, y4)
    check_ir_symbolic_shape(dir_path, target_str, 3)
    os.unsetenv('MS_DEV_SAVE_GRAPHS')
    os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
