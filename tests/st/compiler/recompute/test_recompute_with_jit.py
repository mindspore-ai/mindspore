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
"""Test recomputation with jit."""
import os
import re
import subprocess
import tempfile
import inspect
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore._c_expression import MSContext
from tests.mark_utils import arg_mark

match_dyn_mem = re.compile(
    r'Used peak memory usage \(without fragments\): (.*?)M', re.S)


def get_max(mem_uses):
    max_mem = 0
    for i in mem_uses:
        max_mem = max(max_mem, int(i))
    return max_mem


def run_testcase(testcase_name, expect_memory_usage):
    # Clear log file
    log_filename = testcase_name + ".log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    assert not os.path.exists(log_filename)

    cmd = ("export GLOG_v=1; export MS_ALLOC_CONF=\"memory_recycle:False\"; "
           "export MS_DEV_RUNTIME_CONF=\"ge_kernel:False\"; pytest -s test_recompute.py::") + \
        testcase_name + " > " + log_filename + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_filename)
    with open(log_filename, "r", encoding="utf-8") as f:
        data = f.read()
    mem_uses = re.findall(match_dyn_mem, data)
    assert len(mem_uses) == 2
    max_mem = get_max(mem_uses)
    assert max_mem == expect_memory_usage
    # Clear log file
    os.remove(log_filename)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_recompute_cell_recompute_with_jit1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the cell recompute api and run grad in jit.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_block_recompute_with_jit1", 46)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_recompute_block_recompute_func_api_with_jit1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the recomputed func api and run grad in jit.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_block_recompute_func_api_with_jit1", 46)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_recompute_cell_recompute_with_jit2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the cell recompute api and run grad in jit.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_block_recompute_with_jit2", 45)


def read_file(folder_path, subfix=".ir", contains_str="validate_"):
    _files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(subfix) and contains_str in filename:
            _files.append(os.path.join(folder_path, filename))
    r_file = max(_files, key=os.path.getctime)
    with open((os.path.join(r_file)), 'r', encoding="utf-8") as f:
        content = f.read()
    return content


def check_str_in_validate_ir(saved_graphs_path, check_str_dict):
    content = read_file(saved_graphs_path)
    content = content.replace("need_cse_after_recompute", "IGNORE_TAG")
    auto_monad_op_list = ["Load", "UpdateState", "Depend", "make_tuple"]
    for op in auto_monad_op_list:
        content = re.sub(r"{}[(].*[:]".format(op), "IGNORE_TAG", content)
    flag_list = []
    for key, value in check_str_dict.items():
        match_list = re.findall(r'{}\W(.*)\n'.format(key), content)
        if value in "".join(match_list):
            flag_list.append(True)
        else:
            flag_list.append(False)
    assert all(flag_list)


class NetRelu(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.net(x)
        x = self.relu(x)
        return x


class Conv2dAddReluMean(nn.Cell):
    def __init__(self, has_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, weight_init="ones",
                              bias_init='zeros', has_bias=has_bias)
        self.add = ops.Add()
        self.relu = ops.ReLU()
        self.mean = ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.conv(x)
        x = self.add(x, x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        return x


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_recompute_set_net_cell_true_primitive_attr_false():
    """
    Feature: Net recomputation and Primitive recompuation.
    Description: Enable recompute on network cell, but explicitly disable it on a primitive via add_prim_attr.
    Expectation: The primitive (Conv2D) should have recompute=False in IR, and gradient results match baseline.
    """
    original_save_graphs = os.environ.get("MS_DEV_SAVE_GRAPHS")
    original_save_graphs_path = os.environ.get("MS_DEV_SAVE_GRAPHS_PATH")
    os.environ["MS_DEV_SAVE_GRAPHS"] = "2"

    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class RecomputeNet(NetRelu):
        def __init__(self, net):
            super().__init__(net=net)
            self.net.recompute()
            self.net.conv.conv2d.add_prim_attr("recompute", False)

    net1 = RecomputeNet(Conv2dAddReluMean())
    net2 = NetRelu(Conv2dAddReluMean())
    data = Tensor(np.random.randn(1, 3, 32, 32), dtype=ms.float32)

    out1 = ms.jit(ms.grad(net1))(data)
    check_str_in_validate_ir(
        saved_graphs_path, {"Conv2D": "recompute: Bool(0)"})
    out2 = ms.jit(ms.grad(net2))(data)

    if MSContext.get_instance().get_ascend_soc_version() == 'ascend910':
        np.allclose(out1.asnumpy(), out2.asnumpy(), 0.001, 0.001)
    else:
        np.allclose(out1.asnumpy(), out2.asnumpy())

    if original_save_graphs is not None:
        os.environ["MS_DEV_SAVE_GRAPHS"] = original_save_graphs
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS", None)

    if original_save_graphs_path is not None:
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = original_save_graphs_path
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS_PATH", None)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_recompute_set_net_cell_true_primitive_attr_true():
    """
    Feature: Net recomputation and Primitive recompuation.
    Description: Enable recompute on both network cell and the underlying primitive (Conv2D).
    Expectation: The primitive should have recompute=True in IR, and gradient results match baseline.
    """
    original_save_graphs = os.environ.get("MS_DEV_SAVE_GRAPHS")
    original_save_graphs_path = os.environ.get("MS_DEV_SAVE_GRAPHS_PATH")
    os.environ["MS_DEV_SAVE_GRAPHS"] = "2"

    saved_graphs_path = tempfile.mkdtemp(f"_{inspect.stack()[0].function}")
    os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = saved_graphs_path

    class RecomputeNet(NetRelu):
        def __init__(self, net):
            super().__init__(net=net)
            self.net.recompute()
            self.net.conv.conv2d.recompute()

    data = Tensor(np.random.randn(1, 3, 32, 32), dtype=ms.float32)

    net1 = RecomputeNet(Conv2dAddReluMean())
    net2 = NetRelu(Conv2dAddReluMean())

    out1 = ms.jit(ms.grad(net1))(data)
    check_str_in_validate_ir(
        saved_graphs_path, {"Conv2D": "recompute: Bool(1)"})
    out2 = ms.jit(ms.grad(net2))(data)
    np.allclose(out1.asnumpy(), out2.asnumpy())

    if original_save_graphs is not None:
        os.environ["MS_DEV_SAVE_GRAPHS"] = original_save_graphs
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS", None)

    if original_save_graphs_path is not None:
        os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = original_save_graphs_path
    else:
        os.environ.pop("MS_DEV_SAVE_GRAPHS_PATH", None)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_recompute_set_cell_conv_recompute_true_pynative():
    """
    Feature: Net recomputation and Primitive recompuation.
    Description: Attempt to call recompute() on a primitive within Pynative mode.
    Expectation: A TypeError should be raised.
    """
    ms.runtime.launch_blocking()

    class RecomputeNet(Conv2dAddReluMean):
        def __init__(self):
            super().__init__()
            self.conv.recompute()

    with pytest.raises(TypeError) as info:
        a = RecomputeNet()
        a(1)
    assert "The primitive[Conv2D]'s input arguments" in str(info.value)
