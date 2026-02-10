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
""" test_context """
import os
import mindspore as ms
from mindspore import context

def test_set_deterministic_true():
    """"
    Feature: test set_deterministic setting
    Description: Test case for simplest deterministic setting
    Expectation: The results are as expected
    """
    ms.set_deterministic(True)
    assert os.getenv("HCCL_DETERMINISTIC") == "true"
    assert os.getenv("TE_PARALLEL_COMPILER") == "1"

def test_set_deterministic_strict():
    """"
    Feature: test set_deterministic setting
    Description: Test case for simplest deterministic setting
    Expectation: The results are as expected
    """
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    ms.set_deterministic(True)
    assert os.getenv("HCCL_DETERMINISTIC") == "strict"
    assert os.getenv("TE_PARALLEL_COMPILER") == "1"


def test_set_deterministic_false():
    """"
    Feature: test set_deterministic setting
    Description: Test case for simplest deterministic setting
    Expectation: The results are as expected
    """
    ms.set_deterministic(False)
    assert "HCCL_DETERMINISTIC" not in os.environ
    assert "TE_PARALLEL_COMPILER" not in os.environ

def test_set_device_target_GPU():
    """"
    Feature: test set_device setting
    Description: Test case for GPU
    Expectation: The results are as expected
    """
    ms.set_device("GPU")
    assert context.get_context("device_target") == "GPU"

def test_set_device_target_Ascend():
    """"
    Feature: test set_device setting
    Description: Test case for Ascend
    Expectation: The results are as expected
    """
    ms.set_device("Ascend")
    assert context.get_context("device_target") == "Ascend"

def test_set_device_id():
    """"
    Feature: test set_device setting
    Description: Test case for device id
    Expectation: The results are as expected
    """
    ms.set_device("Ascend", 2)
    assert context.get_context("device_id") == 2
