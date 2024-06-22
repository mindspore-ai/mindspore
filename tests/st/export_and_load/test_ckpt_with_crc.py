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
import stat

import pytest
import os
import mindspore as ms
import mindspore.nn as nn

from mindspore.common.file_system import FileSystem


class Network(nn.Cell):
    def __init__(self, lin_weight, lin_bias):
        super().__init__()
        self.lin = nn.Dense(2, 3, weight_init=lin_weight, bias_init=lin_bias)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.lin(x)
        out = self.relu(out)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ckpt_save_with_crc(mode):
    """
    Feature: Save ckpt with crc check.
    Description: Save ckpt with crc check.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    weight = ms.Tensor([[0.27201429, 2.22499485],
                        [-0.5636731, -2.21354142],
                        [1.3987198, 0.04099071]], dtype=ms.float32)
    bias = ms.Tensor([-0.41271235, 0.28378568, -0.81612898], dtype=ms.float32)
    net = Network(weight, bias)
    ms.save_checkpoint(net, './save_with_crc.ckpt', crc_check=True)

    _ckpt_fs = FileSystem()
    with _ckpt_fs.open("./save_with_crc.ckpt", *_ckpt_fs.open_args) as f:
        pb_content = f.read()
        assert b"crc_num" in pb_content

    ms.load_checkpoint("./save_with_crc.ckpt", crc_check=True)
    ms.load_checkpoint("./save_with_crc.ckpt", crc_check=False)
    os.chmod('./save_with_crc.ckpt', stat.S_IWRITE)
    os.remove('./save_with_crc.ckpt')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ckpt_save_with_crc_failed(mode):
    """
    Feature: Save ckpt with crc check.
    Description: Save ckpt with crc check.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    weight = ms.Tensor([[0.27201429, 2.22499485],
                        [-0.5636731, -2.21354142],
                        [1.3987198, 0.04099071]], dtype=ms.float32)
    bias = ms.Tensor([-0.41271235, 0.28378568, -0.81612898], dtype=ms.float32)
    net = Network(weight, bias)
    ms.save_checkpoint(net, './save_with_crc_failed.ckpt', crc_check=True)

    _ckpt_fs = FileSystem()
    os.chmod("./save_with_crc_failed.ckpt", stat.S_IWRITE)
    with _ckpt_fs.open("./save_with_crc_failed.ckpt", *_ckpt_fs.create_args) as f:
        f.write(b"111")

    with pytest.raises(ValueError):
        ms.load_checkpoint("./save_with_crc_failed.ckpt", crc_check=True)
    os.chmod('./save_with_crc_failed.ckpt', stat.S_IWRITE)
    os.remove('./save_with_crc_failed.ckpt')
