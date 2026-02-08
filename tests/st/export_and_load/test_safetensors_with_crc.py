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
import os
import time
import stat
import mindspore as ms
import mindspore.nn as nn

from tests.mark_utils import arg_mark


def remove_ckpt(file_name):
    """remove ckpt."""
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.00)
def test_save_and_load_sft_with_crc():
    """
    Feature: save and load checkpoint
    Description: test ms.save_checkpoint and ms.load_checkpoint with crc_check
    Expectation: success
    """
    net = nn.Dense(2, 2)
    ckpt_path = "checkpoint_1.safetensors"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net.parameters_dict(), ckpt_path, format="safetensors", crc_check=True)
    ms.load_checkpoint(ckpt_path, format="safetensors", crc_check=True)
    remove_ckpt(ckpt_path)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_save_and_load_sft_with_async():
    """
    Feature: save and load checkpoint
    Description: test ms.save_checkpoint and ms.load_checkpoint with crc_check
    Expectation: success
    """
    net = nn.Dense(2, 2)
    ckpt_path = "checkpoint_async.safetensors"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net.parameters_dict(), ckpt_path, format="safetensors", async_save=True)
    time.sleep(6)
    assert os.path.exists(ckpt_path)
    remove_ckpt(ckpt_path)
