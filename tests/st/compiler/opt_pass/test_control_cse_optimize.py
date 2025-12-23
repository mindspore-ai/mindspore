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
"""Test ces optimize pass"""
import numpy as np
import pytest
import inspect
import tempfile
import os
import re
import mindspore as ms
from mindspore import nn
from mindspore.nn import Cell
import mindspore.ops.operations as op
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def setup_env():
    original_save_graphs = os.environ.get("MS_DEV_SAVE_GRAPHS")
    original_save_graphs_path = os.environ.get("MS_DEV_SAVE_GRAPHS_PATH")

    os.environ["MS_DEV_SAVE_GRAPHS"] = "2"

    yield

    if original_save_graphs is not None:
        os.environ["MS_DEV_SAVE_GRAPHS"] = original_save_graphs
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS", None)

    if original_save_graphs_path is not None:
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = original_save_graphs_path
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS_PATH", None)


def find_newest_validateir_file(folder_path):
    ir_files = map(lambda f: os.path.join(folder_path, f),
                   filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                          os.listdir(folder_path)))
    return max(ir_files, key=os.path.getctime)


def read_file(saved_graphs_path):
    content = ''
    filename = find_newest_validateir_file(saved_graphs_path)
    with open((os.path.join(filename)), 'r', encoding="utf-8") as f:
        content = f.read()
    return content


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_if():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE correctly handles Conv2D under if-else control flow.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=100)
            self.flatten = nn.Flatten()
            self.sigmoid = op.Sigmoid()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            conv = self.conv(x)  # (16, 3, 28, 28)
            if self.a > self.b:
                tmp1 = self.sigmoid(conv)
            else:
                tmp1 = conv
            if self.b < self.c or self.a > self.c:
                tmp2 = self.relu(tmp1)
            else:
                tmp2 = tmp1
            flatten = self.flatten(tmp2)  # 16, 2352
            dense = self.dense(flatten)
            return dense

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 1, 32, 32), dtype=ms.float32)
    ms.jit(net)(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_while():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE preserves single Conv2D instance despite while loop.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=100)
            self.flatten = nn.Flatten()
            self.sigmoid = op.Sigmoid()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 3, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            conv = self.conv(x)  # (16, 3, 28, 28)
            while self.a > self.b:
                conv = self.sigmoid(conv)
                self.b = self.b + 1

            flatten = self.flatten(conv)  # 16, 2352
            dense = self.dense(flatten)
            return dense

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 1, 32, 32), dtype=ms.float32)
    ms.jit(net, backend="ms_backend")(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_if_for():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE works with nested if and for loops.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=100)
            self.flatten = nn.Flatten()
            self.sigmoid = op.Sigmoid()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 3, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")

        def construct(self, x):
            conv = self.conv(x)  # (16, 3, 28, 28)
            if self.a > self.b:
                for _ in range(0, 2):
                    conv = self.relu(conv)
            else:
                for _ in range(0, 3):
                    conv = self.sigmoid(conv)

            flatten = self.flatten(conv)  # 16, 2352
            dense = self.dense(flatten)
            return dense

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 1, 32, 32), dtype=ms.float32)
    ms.jit(net)(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_func():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE handles calls to internal methods with control flow.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=100)
            self.flatten = nn.Flatten()
            self.sigmoid = op.Sigmoid()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 3, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def func1(self, x):
            if self.a > self.b:
                out = self.relu(x)
            else:
                out = self.sigmoid(x)
            return out

        def func2(self, x):
            for _ in range(0, 2):
                x = self.add(x, x)
            return x

        def construct(self, x):
            conv = self.conv(x)  # (16, 3, 28, 28)
            if self.b > self.c:
                tmp1 = self.func1(conv)
            else:
                tmp1 = self.func2(conv)

            flatten = self.flatten(tmp1)  # 16, 2352
            dense = self.dense(flatten)
            return dense

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 1, 32, 32), dtype=ms.float32)
    ms.jit(net)(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_subnet():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE works across subnetworks and method calls with control flow.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class SubNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.tanh = op.Tanh()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")

        def construct(self, x):
            if self.a > self.b:
                x = self.tanh(x)
            else:
                x = self.relu(x)
            return x

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.subnet = SubNet()
            self.relu = op.ReLU()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=100)
            self.flatten = nn.Flatten()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 3, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def func(self, x):
            for i in range(0, 2):
                x = self.add(x, x) + i
            print(x)
            return x

        def construct(self, x):
            conv = self.conv(x)  # (16, 3, 28, 28)
            if self.b < self.c:
                tmp1 = self.func(conv)
            else:
                tmp1 = conv

            for _ in range(0, 2):
                tmp1 = self.subnet(tmp1)

            flatten = self.flatten(tmp1)  # 16, 2352
            dense = self.dense(flatten)
            return dense

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 1, 32, 32), dtype=ms.float32)
    ms.jit(net)(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_subnet_and_sink():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE handles multi-input subnet and data sink pattern.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class SubNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.tanh = op.Tanh()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")

        def construct(self, x):
            if self.a > self.b:
                x = self.tanh(x)
            else:
                x = self.relu(x)
            return x

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.subnet = SubNet()
            self.relu = op.ReLU()
            self.conv = nn.Conv2d(in_channels=3,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=12)
            self.flatten = nn.Flatten()
            self.add = op.TensorAdd()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 3, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def func(self, x):
            for i in range(0, 2):
                x = self.add(x, x) + i
            return x

        def construct(self, x):
            conv = self.conv(x)  # (16, 3, 28, 28)
            if self.b < self.c:
                tmp1 = self.func(conv)
            else:
                tmp1 = conv

            for _ in range(0, 2):
                tmp1 = self.subnet(tmp1)

            flatten = self.flatten(tmp1)  # 16, 2352
            dense = self.dense(flatten)
            return dense

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 3, 32, 32), dtype=ms.float32)
    ms.jit(net)(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_heterogeneous_and_parameter():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE works with heterogeneous primitive targets and parameters.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=100)
            self.flatten = nn.Flatten()
            self.add = op.TensorAdd().add_prim_attr('primitive_target', 'CPU')
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 3, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")
            add_np = np.full((16, 3, 28, 28), 5, dtype=np.float32)
            self.add_param = Parameter(Tensor(add_np), name="add_param")

        def func(self, x):
            for _ in range(0, 2):
                x = self.add(x, self.add_param)
            return x

        def construct(self, x):
            conv = self.conv(x)  # (16, 3, 28, 28)
            if self.b < self.c:
                tmp1 = self.func(conv)
            else:
                tmp1 = self.relu(conv)

            flatten = self.flatten(tmp1)  # 16, 2352
            dense = self.dense(flatten)
            return dense

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 1, 32, 32), dtype=ms.float32)
    ms.jit(net, backend="ms_backend")(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctrl_cse_optimize_with_dynamic_shape():
    """
    Feature: Control-flow aware CSE optimization.
    Description: CSE handles dynamic shape operations under control flow.
    Expectation: Conv2D appears exactly once in the optimized IR.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = op.ReLU()
            self.sigmoid = op.Sigmoid()
            self.add = op.TensorAdd()
            self.expanddims1 = op.ExpandDims()
            self.expanddims2 = op.ExpandDims()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=3,
                                  kernel_size=5,
                                  pad_mode='valid',
                                  weight_init='ones',
                                  bias_init='ones',
                                  has_bias=True)
            self.dense = nn.Dense(in_channels=2352, out_channels=100)
            self.flatten = nn.Flatten()
            a = np.full((1,), 5, dtype=np.float32)
            self.a = Parameter(Tensor(a), name="a")
            b = np.full((1,), 4, dtype=np.float32)
            self.b = Parameter(Tensor(b), name="b")
            c = np.full((1,), 7, dtype=np.float32)
            self.c = Parameter(Tensor(c), name="c")

        def construct(self, x):
            conv = self.conv(x)
            if self.a > self.b:
                i = 1
            else:
                i = 2
            if self.b < self.c:
                i = self.expanddims1(conv, i)
            else:
                i = self.expanddims2(conv, i)
            relu = self.relu(i)
            flatten = self.flatten(relu)
            out = self.dense(flatten)
            return out

    net = Net()
    fake_data = ms.Tensor(np.random.randn(1, 1, 32, 32), dtype=ms.float32)
    ms.jit(net)(fake_data)
    content = read_file(saved_graphs_path)
    conv2d_num = re.findall(r'Conv2D\(', content)
    assert len(conv2d_num) == 1
