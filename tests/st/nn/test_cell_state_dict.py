# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
""" test cell buffers"""
import os
import stat
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import _Buffer


class SimpleNet(nn.Cell):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.buffer_0 = _Buffer(Tensor(np.array([1, 2, 3]).astype(np.float32)))
        self.param_0 = ms.Parameter(Tensor(np.array([10, 20, 30]).astype(np.float32)))
        self.dense = nn.Dense(5, 3)

    def construct(self, x):
        return (self.dense(x) + self.buffer_0) * self.buffer_1 + self.param_0


class ComplexNet(nn.Cell):
    def __init__(self, sub_net):
        super(ComplexNet, self).__init__()
        self.sub_net = sub_net
        self.buffer_1 = _Buffer(Tensor(np.array([6, 6, 6]).astype(np.float32)))
        self.register_buffer('buffer_2', Tensor(np.array([7, 7, 7]).astype(np.float32)))
        self.param_1 = ms.Parameter(Tensor(np.array([8, 8, 8]).astype(np.float32)))

    def construct(self, x):
        return self.sub_net(x) + self.buffer_1 - self.buffer_2 - self.param_1


def remove_ckpt(file_name):
    """remove ckpt."""
    if os.path.exists(file_name) and file_name.endswith(".ckpt"):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net_state_dict(mode):
    """
    Feature: test state_dict of net
    Description: Verify the result of state_dict
    Expectation: success
    """
    ms.set_context(mode=mode)
    net1 = SimpleNet()
    net0 = ComplexNet(net1)
    ms.save_checkpoint(net0.state_dict(), f"./ms_net0_state_dict{mode}.ckpt")

    new_net1 = SimpleNet()
    new_net0 = ComplexNet(new_net1)
    new_net0.load_state_dict(ms.load_checkpoint(f"./ms_net0_state_dict{mode}.ckpt"))

    remove_ckpt(f"./ms_net0_state_dict{mode}.ckpt")

    old_names = list(net0.state_dict().keys())
    new_names = list(new_net0.state_dict().keys())
    for i, name in enumerate(old_names):
        assert new_names[i] == name

    old_values = list(net0.state_dict().values())
    new_values = list(new_net0.state_dict().values())
    for i, value in enumerate(old_values):
        if isinstance(value, ms.Parameter):
            old_value = value.value()
            new_value = new_values[i].value()
            assert np.allclose(old_value.asnumpy(), new_value.asnumpy())
        else:
            assert np.allclose(value.asnumpy(), new_values[i].asnumpy())


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net_state_dict_hook(mode):
    """
    Feature: test state_dict hooks of net
    Description: Verify the result of state_dict hooks
    Expectation: success
    """
    ms.set_context(mode=mode)

    def state_dict_pre_hook(cell, prefix, keep_vars):
        cell.set_label_state_dict_pre_hook = "state_dict pre hook success"

    def state_dict_post_hook(cell, state_dict, prefix, local_metadata):
        cell.set_label_state_dict_post_hook = "state_dict post hook success"

    def load_state_dict_pre_hook(cell, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                 error_msgs):
        cell.set_label_load_state_dict_pre_hook = "load state_dict pre hook success"

    def load_state_dict_post_hook(cell, incompatible_keys):
        cell.set_label_load_state_dict_post_hook = "load state_dict post hook success"

    net = SimpleNet()

    net.register_state_dict_pre_hook(state_dict_pre_hook)
    net.register_state_dict_post_hook(state_dict_post_hook)

    ms.save_checkpoint(net.state_dict(), f"./ms_hooks_model{mode}.ckpt")

    assert net.set_label_state_dict_pre_hook == "state_dict pre hook success"
    assert net.set_label_state_dict_pre_hook == "state_dict pre hook success"

    new_net = SimpleNet()
    new_net.register_load_state_dict_pre_hook(load_state_dict_pre_hook)
    new_net.register_load_state_dict_post_hook(load_state_dict_post_hook)

    new_net.load_state_dict(ms.load_checkpoint(f"./ms_hooks_model{mode}.ckpt"))

    remove_ckpt(f"./ms_hooks_model{mode}.ckpt")

    assert new_net.set_label_load_state_dict_pre_hook == "load state_dict pre hook success"
    assert new_net.set_label_load_state_dict_post_hook == "load state_dict post hook success"
