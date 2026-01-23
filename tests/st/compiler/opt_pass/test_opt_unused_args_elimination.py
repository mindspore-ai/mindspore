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
# ==============================================================================
"""Test optimization of unused args elimination."""

import inspect
import os
import re
import pytest
import tempfile
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore import nn, lazy_inline, Tensor, Parameter
import numpy as np
import torch
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def setup_env():
    original_save_graphs = os.environ.get("MS_DEV_SAVE_GRAPHS")
    original_save_graphs_path = os.environ.get("MS_DEV_SAVE_GRAPHS_PATH")

    os.environ["MS_DEV_SAVE_GRAPHS"] = "1"

    yield

    if original_save_graphs is not None:
        os.environ["MS_DEV_SAVE_GRAPHS"] = original_save_graphs
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS", None)

    if original_save_graphs_path is not None:
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = original_save_graphs_path
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS_PATH", None)


def search_string(file_name, string_to_search):
    with open(file_name, 'r', encoding="utf-8") as read_obj:
        return [(idx + 1, line.rstrip()) for idx, line in enumerate(read_obj) if string_to_search in line]


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def check_partial_args(saved_graphs_path, idx=0, expect=1):
    irfile = find_newest_validateir_file(saved_graphs_path)
    partial_line = search_string(irfile, "= Partial")
    partial_args_num = partial_line[idx][1].count('%') - 1
    assert partial_args_num == expect, "Unused args not eliminated"


class Net1a(Cell):
    def __init__(self):
        super().__init__()
        self.t = 2

    def construct(self, x, y, z):
        if x > 1:
            out = self.func1(y, z)
        else:
            out = z
        return out

    def func1(self, a, b):
        b = b + b
        return self.t * a


class Net1b(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = 2

    def forward(self, x, y, z):
        if x > 1:
            out = self.func1(y, z)
        else:
            out = z
        return out

    def func1(self, a, b):
        b = b + b
        return self.t * a


def ms2torch(tensor):
    tnp = tensor.asnumpy()
    x = torch.tensor(tnp, dtype=torch.float)
    return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parse_opt_unused_args():
    """
    Feature: Unused argument elimination.
    Description: Eliminate unused formal parameter in called function.
    Expectation: Partial call retains only 1 used argument.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path
    net1 = Net1a()
    x = Tensor([2], mstype.float32)
    y = Tensor(np.random.rand(2, 3), mstype.float32)
    z = Tensor(np.random.rand(2, 3), mstype.float32)
    out1 = ms.jit(net1)(x, y, z)
    net2 = Net1b()
    out2 = net2(ms2torch(x), ms2torch(y), ms2torch(z))
    np.allclose(out2.numpy(), out1.asnumpy(), 0.0001, 0.0001)
    check_partial_args(saved_graphs_path, expect=1)
    del os.environ["MS_DEV_SAVE_GRAPHS_PATH"]


class Net2a(Cell):
    def __init__(self):
        super().__init__()
        self.t = 2

    def construct(self, x, y, z):
        if x > 1:
            out = self.func1(y, z, y, z)
        else:
            out = self.func1(z, y, z, y)
        return out

    def func1(self, a, b, c, d):
        _ = b + c
        return self.t * a + 3 * d


class Net2b(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = 2

    def forward(self, x, y, z):
        if x > 1:
            out = self.func1(y, z, y, z)
        else:
            out = self.func1(z, y, z, y)
        return out

    def func1(self, a, b, c, d):
        _ = b + c
        return self.t * a + 3 * d


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parse_opt_unused_formal_args():
    """
    Feature: Unused argument elimination.
    Description: Eliminate unused formal parameters among multiple arguments.
    Expectation: Partial call retains only 2 used arguments.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path
    net1 = Net2a()
    x = Tensor([2], mstype.float32)
    y = Tensor(np.random.rand(2, 3), mstype.float32)
    z = Tensor(np.random.rand(2, 3), mstype.float32)
    out1 = ms.jit(net1)(x, y, z)
    net2 = Net2b()
    out2 = net2(ms2torch(x), ms2torch(y), ms2torch(z))
    np.allclose(out2.numpy(), out1.asnumpy(), 0.0001, 0.0001)
    check_partial_args(saved_graphs_path, expect=2)
    del os.environ["MS_DEV_SAVE_GRAPHS_PATH"]


class Net3a(Cell):
    def __init__(self):
        super().__init__()
        self.t = 1

    def construct(self, x, y, z):
        if x > 0:
            out = self.func1(x, y, z)
        else:
            out = z + z
        return out

    def func1(self, x, a, b):
        return self.func2(x, a, b)

    def func2(self, x, a, b):
        if x > self.t:
            out = a + b
        else:
            out = b
        return out


class Net3b(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = 1

    def forward(self, x, y, z):
        if x > 0:
            out = self.func1(x, y, z)
        else:
            out = z + z
        return out

    def func1(self, x, a, b):
        return self.func2(x, a, b)

    def func2(self, x, a, b):
        if x > self.t:
            return a + b
        return b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parse_opt_unused_args_two_func():
    """
    Feature: Unused argument elimination.
    Description: Eliminate unused args across nested function calls.
    Expectation: Partial call at deeper level retains 2 used arguments.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path
    net1 = Net3a()
    x = Tensor([2], mstype.float32)
    y = Tensor(np.random.rand(2, 3), mstype.float32)
    z = Tensor(np.random.rand(2, 3), mstype.float32)
    out1 = ms.jit(net1)(x, y, z)
    net2 = Net3b()
    out2 = net2(ms2torch(x), ms2torch(y), ms2torch(z))
    np.allclose(out2.numpy(), out1.asnumpy(), 0.0001, 0.0001)
    check_partial_args(saved_graphs_path, idx=2, expect=2)
    del os.environ["MS_DEV_SAVE_GRAPHS_PATH"]


class Net4a(Cell):
    def __init__(self, param):
        super().__init__()
        self.t = 2
        self.p = Parameter(Tensor(param), name='p')

    def construct(self, x, y, z):
        if x > 1:
            out = self.func1(y, z)
        else:
            out = z * z
        return out

    def func1(self, a, b):
        a = b
        b = a
        return self.p


class Net4b(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.t = 2
        self.p = torch.nn.parameter.Parameter(
            torch.tensor(param, dtype=torch.float32))

    def forward(self, x, y, z):
        if x > 1:
            out = self.func1(y, z)
        else:
            out = z * z
        return out

    def func1(self, a, b):
        a = b
        b = a
        return self.p


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parse_opt_unused_args_return_param():
    """
    Feature: Unused argument elimination.
    Description: Eliminate unused args when function returns a Parameter.
    Expectation: Partial call retains only 1 used argument.
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path
    param = np.random.rand(2, 3).astype(np.float32)
    net1 = Net4a(param)
    x = Tensor([2], mstype.float32)
    y = Tensor(np.random.rand(2, 3), mstype.float32)
    z = Tensor(np.random.rand(2, 3), mstype.float32)
    out1 = ms.jit(net1)(x, y, z)

    net2 = Net4b(param)
    out2 = net2(ms2torch(x), ms2torch(y), ms2torch(z))
    np.allclose(out2.detach().numpy(),
                out1.asnumpy(), 0.0001, 0.0001)
    check_partial_args(saved_graphs_path, expect=1)
    del os.environ["MS_DEV_SAVE_GRAPHS_PATH"]


class Block(Cell):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm((128, 16, 32),
                                       begin_norm_axis=1, begin_params_axis=1)

    def construct(self, x):
        x = self.layer_norm(x)
        return x


class OuterBlock(Cell):
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.block = Block()

    def construct(self, x):
        return self.block(x)


class PNet(Cell):
    def __init__(self):
        super().__init__()
        self.blocks = nn.CellList()
        for _ in range(3):
            b = OuterBlock()
            self.blocks.append(b)

    def construct(self, x):
        out = x
        for i in range(3):
            out = self.blocks[i](out)
        return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
def test_parse_opt_unused_args_lazy_inline():
    """
    Feature: Unused argument elimination.
    Description: Eliminate unused args in lazy-inline nested Cell structures.
    Expectation: Partial call retains 4 used arguments (e.g., self and inputs).
    """
    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path
    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = PNet()
    grad_net = ms.grad(net)
    ms.jit(grad_net)(x)
    check_partial_args(saved_graphs_path, expect=4)
    del os.environ["MS_DEV_SAVE_GRAPHS_PATH"]
