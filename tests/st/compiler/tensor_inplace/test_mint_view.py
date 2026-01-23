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
"""Test view mint api."""
# pylint: disable=W1514
import os
import pytest
import mindspore as ms
from mindspore import Tensor, mint, nn, context, ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_flatten_mint():
    """
    Feature: test flatten mint api.
    Description: test flatten mint api.
    Expectation: success.
    """

    file_name = "test_mint_view_inner.py"
    log_file_name = "test_mint_view_flatten.log"
    function_name = "::test_flatten_mint_innier"
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, file_name)
    assert os.path.exists(file_name)

    log_file_name = os.path.join(_cur_dir, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)

    cmd_first = "GLOG_v=2 pytest -s " + file_name + function_name + " > " + log_file_name + " 2>&1"
    out = os.popen(cmd_first)
    out.read()

    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()
    assert "The mint interface flatten was called" in data_first
    os.remove(log_file_name)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_imag_mint():
    """
    Feature: test imag mint api.
    Description: test imag mint api.
    Expectation: success.
    """

    class ImagMintNet(nn.Cell):
        def construct(self, x):
            out = mint.imag(x)
            return out

    with pytest.raises(RuntimeError) as info:
        context.set_context(mode=context.GRAPH_MODE)
        x = ms.tensor([0.22 + 0.3511j, -0.55 - 0.6796j, -1.92 - 0.033j, -0.38 - 0.2991j])
        net = ImagMintNet()
        net(x)
    assert "RealViewFuncImpl::InferType is not implemented,because it is not supported in Graph mode" in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',essential_mark='essential')
def test_real_mint():
    """
    Feature: test real mint api.
    Description: test real mint api.
    Expectation: success.
    """
    class RealMintNet(nn.Cell):
        def construct(self, x):
            out = mint.real(x)
            return out

    with pytest.raises(RuntimeError) as info:
        context.set_context(mode=context.GRAPH_MODE)
        real = Tensor([1.1, 2.1, 3.1], ms.float32)
        imag = Tensor([4.1, 5.1, 6.1], ms.float32)
        x = ops.Complex()(real, imag)
        net = RealMintNet()
        net(x)
    assert "RealViewFuncImpl::InferType is not implemented,because it is not supported in Graph mode" in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reshape_mint():
    """
    Feature: test reshape mint api.
    Description: test reshape mint api.
    Expectation: success.
    """

    file_name = "test_mint_view_inner.py"
    log_file_name = "test_mint_view_reshape.log"
    function_name = "::test_reshape_mint_inner"
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, file_name)
    assert os.path.exists(file_name)

    log_file_name = os.path.join(_cur_dir, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)

    cmd_first = "GLOG_v=2 pytest -s " + file_name + function_name + " > " + log_file_name + " 2>&1"
    out = os.popen(cmd_first)
    out.read()

    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()
    assert "The mint interface reshape was called" in data_first
    os.remove(log_file_name)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_squeeze_mint():
    """
    Feature: test squeeze mint api.
    Description: test squeeze mint api.
    Expectation: success.
    """

    file_name = "test_mint_view_inner.py"
    log_file_name = "test_mint_view_squeeze.log"
    function_name = "::test_squeeze_mint_inner"
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, file_name)
    assert os.path.exists(file_name)

    log_file_name = os.path.join(_cur_dir, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)

    cmd_first = "GLOG_v=2 pytest -s " + file_name + function_name + " > " + log_file_name + " 2>&1"
    out = os.popen(cmd_first)
    out.read()

    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()
    assert "The mint interface squeeze was called" in data_first
    os.remove(log_file_name)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_t_mint():
    """
    Feature: test t mint api.
    Description: test t mint api.
    Expectation: success.
    """

    file_name = "test_mint_view_inner.py"
    log_file_name = "test_mint_view_t.log"
    function_name = "::test_t_mint_inner"
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, file_name)
    assert os.path.exists(file_name)

    log_file_name = os.path.join(_cur_dir, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)

    cmd_first = "GLOG_v=2 pytest -s " + file_name + function_name + " > " + log_file_name + " 2>&1"
    out = os.popen(cmd_first)
    out.read()

    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()
    assert "The mint interface t was called" in data_first
    os.remove(log_file_name)
