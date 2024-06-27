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
import pytest
import mindspore as ms
import mindspore.nn as nn
from safetensors.numpy import save_file
import stat


def create_ckpt_directory(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for rank in range(8):
        rank_dir = os.path.join(base_dir, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        net = nn.Dense(2, 2)
        ckpt_path = os.path.join(rank_dir, f"checkpoint_{rank}.ckpt")
        ms.save_checkpoint(net.parameters_dict(), ckpt_path)
    return base_dir


def create_safetensors_directory(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for rank in range(8):
        rank_dir = os.path.join(base_dir, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        net = nn.Dense(2, 2)
        params = net.parameters_dict()
        param_dict = {name: param.asnumpy() for name, param in params.items()}
        safetensors_path = os.path.join(rank_dir, f"checkpoint_{rank}.safetensors")
        save_file(param_dict, safetensors_path)
    return base_dir


def cleanup_directory(directory):
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                file_name = os.path.join(root, name)
                os.chmod(file_name, stat.S_IWUSR)
                os.remove(file_name)
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(directory)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ckpt_to_safetensors():
    """
    Feature: ckpt_to_safetensors
    Description: test ms.ckpt_to_safetensors
    Expectation: success
    """
    base_dir = create_ckpt_directory("test_ckpt_dir1")
    save_path = "test_safetensors_dir1"

    ms.ckpt_to_safetensors(file_path=base_dir, save_path=save_path)

    for rank in range(8):
        rank_dir = os.path.join(save_path, f"rank{rank}")
        assert os.path.exists(rank_dir)
        safetensors_files = [f for f in os.listdir(rank_dir) if f.endswith(".safetensors")]
        assert len(safetensors_files) == 1

    # Clean up
    cleanup_directory(base_dir)
    cleanup_directory(save_path)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_safetensors_to_ckpt():
    """
    Feature: safetensors_to_ckpt
    Description: test ms.safetensors_to_ckpt
    Expectation: success
    """
    base_dir = create_safetensors_directory("test_safetensors_dir2")
    save_path = "test_ckpt_dir2"

    ms.safetensors_to_ckpt(file_path=base_dir, save_path=save_path)

    for rank in range(8):
        rank_dir = os.path.join(save_path, f"rank{rank}")
        assert os.path.exists(rank_dir)
        ckpt_files = [f for f in os.listdir(rank_dir) if f.endswith(".ckpt")]
        assert len(ckpt_files) == 1

    # Clean up
    cleanup_directory(base_dir)
    cleanup_directory(save_path)
