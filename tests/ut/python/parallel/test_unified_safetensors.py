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
import pytest
from mindspore.parallel.transform_safetensors import _find_most_matching_file


def test_find_most_matching_file_none_suffix():
    """
    Feature: Test _find_most_matching_file with None file_suffix.
    Description: When file_suffix is None, should sort by last two numbers and return the last one.
    Expectation: Returns the file with the largest last two numbers.
    """
    rank_ckpts = [
        "model_1_2.safetensors",
        "model_3_4.safetensors",
        "model_2_1.safetensors",
        "model_1_3.safetensors"
    ]
    result = _find_most_matching_file(rank_ckpts, None, "safetensors")
    assert result == "model_3_4.safetensors"


def test_find_most_matching_file_pattern1_match():
    """
    Feature: Test _find_most_matching_file with pattern.
    Description: Test matching files with pattern _X-Y_Z.
    Expectation: Returns the matching file without rank prefix.
    """
    rank_ckpts = [
        "model_rank_1_1-2_3.safetensors",
        "model_rank_1-2_3.safetensors",
    ]
    file_suffix = "_1-2_3"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "model_rank_1_1-2_3.safetensors"
    file_suffix = "1-2_3"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "model_rank_1_1-2_3.safetensors"
    rank_ckpts = [
        "llama_rank_200-200_1.safetensors",
        "llama_rank_200-1200_1.safetensors",
    ]
    file_suffix = "200_1"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "llama_rank_200-200_1.safetensors"

    rank_ckpts = [
        "llama_rank_200-200_1.safetensors",
        "llama_rank_200_1-200_1.safetensors",
    ]
    file_suffix = "200_1"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "llama_rank_200-200_1.safetensors"

    rank_ckpts = [
        "llama_rank_200-200_1.safetensors",
        "llama_rank_200_1-200_1.safetensors",
    ]
    file_suffix = "-200_1"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "llama_rank_200-200_1.safetensors"

    rank_ckpts = [
        "llama_rank_200_1-200_1.safetensors",
        "llama_rank_200_2-200_1.safetensors",
    ]
    file_suffix = "-200_1"
    with pytest.raises(ValueError):
        _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")

    rank_ckpts = [
        "llama_rank_200-200_1.safetensors",
        "llama_rank_200_200-200_1.safetensors",
    ]
    file_suffix = "_200-200_1"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "llama_rank_200_200-200_1.safetensors"

    rank_ckpts = [
        "llama_rank_200-200_1.safetensors",
        "llama_rank_200_200-200_1.safetensors",
        "llama_rank_200_1200-200_1.safetensors"
    ]
    file_suffix = "200-200_1"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "llama_rank_200_200-200_1.safetensors"

    rank_ckpts = [
        "llama_rank_200-200_20.safetensors",
        "llama_rank_200-1200_220.safetensors",
    ]
    file_suffix = "20"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "llama_rank_200-200_20.safetensors"

    rank_ckpts = [
        "llama_rank_200-200_20.safetensors",
        "llama_rank_200-1200_220.safetensors",
    ]
    file_suffix = "2000"
    with pytest.raises(ValueError):
        _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")


def test_find_most_matching_file_pattern3_match():
    """
    Feature: Test _find_most_matching_file with pattern.
    Description: Test matching files with pattern X_Y.
    Expectation: Returns the matching file with - prefix.
    """
    rank_ckpts = [
        "model-1_2.safetensors",
        "model_1-1_2.safetensors",
    ]
    file_suffix = "1_2"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "model-1_2.safetensors"
    file_suffix = "-1_2"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "model-1_2.safetensors"


def test_find_most_matching_file_pattern5_match():
    """
    Feature: Test _find_most_matching_file with pattern.
    Description: Test matching files with pattern Y.
    Expectation: Returns the matching file with - prefix.
    """
    rank_ckpts = [
        "model-1_2.safetensors",
        "model_1-1_2.safetensors",
    ]
    file_suffix = "_2"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "model-1_2.safetensors"
    file_suffix = "2"
    result = _find_most_matching_file(rank_ckpts, file_suffix, "safetensors")
    assert result == "model-1_2.safetensors"
