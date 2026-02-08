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
import mindspore as ms
import mindspore.nn as nn


def create_file(base_dir, file_format):
    """Create weight dir and file."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for rank in range(2):
        rank_dir = os.path.join(base_dir, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        net = nn.Dense(2, 2)
        ckpt_path = os.path.join(rank_dir, f"checkpoint_{rank}.{file_format}")
        ms.save_checkpoint(net, ckpt_path, format=file_format)
    return base_dir


def cleanup_directory(directory):
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                file_name = os.path.join(root, name)
                os.remove(file_name)
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(directory)


def test_ckpt_to_safetensors():
    """
    Feature: ckpt_to_safetensors
    Description: test ms.ckpt_to_safetensors
    Expectation: success
    """
    base_dir = create_file("test_ckpt_dir1", file_format="ckpt")
    save_path = "test_safetensors_dir1"

    ms.ckpt_to_safetensors(file_path=base_dir, save_path=save_path)

    for rank in range(2):
        rank_dir = os.path.join(save_path, f"rank{rank}")
        assert os.path.exists(rank_dir)
        safetensors_files = [f for f in os.listdir(rank_dir) if f.endswith(".safetensors")]
        assert len(safetensors_files) == 1

    # Clean up
    cleanup_directory(base_dir)
    cleanup_directory(save_path)


def test_safetensors_to_ckpt():
    """
    Feature: safetensors_to_ckpt
    Description: test ms.safetensors_to_ckpt
    Expectation: success
    """
    base_dir = create_file("test_safetensors_dir1", file_format="safetensors")
    save_path = "test_ckpt_dir2"

    ms.safetensors_to_ckpt(file_path=base_dir, save_path=save_path)

    for rank in range(2):
        rank_dir = os.path.join(save_path, f"rank{rank}")
        assert os.path.exists(rank_dir)
        ckpt_files = [f for f in os.listdir(rank_dir) if f.endswith(".ckpt")]
        assert len(ckpt_files) == 1

    # Clean up
    cleanup_directory(base_dir)
    cleanup_directory(save_path)


def test_ckpt_to_safetensors_for_single_file():
    """
    Feature: ckpt_to_safetensors for single file
    Description: test ms.ckpt_to_safetensors args single file.
    Expectation: success
    """
    net = nn.Dense(2, 2)
    ms.save_checkpoint(net, "ckpt_weight", format="ckpt")
    ms.ckpt_to_safetensors(file_path="ckpt_weight.ckpt")
    assert os.path.exists("ckpt_weight.safetensors")
    os.remove("ckpt_weight.safetensors")
    os.remove("ckpt_weight.ckpt")

    ms.save_checkpoint(net, "safetensors_weight", format="safetensors")
    ms.safetensors_to_ckpt(file_path="safetensors_weight.safetensors")
    assert os.path.exists("safetensors_weight.ckpt")
    os.remove("safetensors_weight.safetensors")
    os.remove("safetensors_weight.ckpt")


def test_safetensors_to_ckpt_name_map():
    """
    Feature: safetensors_to_ckpt name map
    Description: test ms.safetensors_to_ckpt args name_map
    Expectation: success
    """
    base_dir = create_file("test_safetensors_dir2", file_format="safetensors")
    save_path = "test_ckpt_dir3"

    ms.safetensors_to_ckpt(file_path=base_dir, save_path=save_path, name_map={"weight": "weight_2"})

    for rank in range(2):
        rank_dir = os.path.join(save_path, f"rank{rank}")
        assert os.path.exists(rank_dir)
        ckpt_files = [f for f in os.listdir(rank_dir) if f.endswith(".ckpt")]
        assert len(ckpt_files) == 1

    cleanup_directory(base_dir)
    cleanup_directory(save_path)


def test_ckpt_to_safetensors_file_name_regex():
    """
    Feature: ckpt_to_safetensors file_name_regex
    Description: test ms.ckpt_to_safetensors args file_name_regex
    Expectation: success
    """
    base_dir = create_file("test_ckpt_dir2", file_format="ckpt")
    save_path = "test_safetensors_dir1"

    ms.ckpt_to_safetensors(file_path=base_dir, save_path=save_path, file_name_regex="checkpoint")

    for rank in range(2):
        rank_dir = os.path.join(save_path, f"rank{rank}")
        assert os.path.exists(rank_dir)
        safetensors_files = [f for f in os.listdir(rank_dir) if f.endswith(".safetensors")]
        assert len(safetensors_files) == 1

    cleanup_directory(base_dir)
    cleanup_directory(save_path)
