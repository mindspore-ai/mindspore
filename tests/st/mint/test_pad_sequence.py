# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Tests for mint.pad_sequence input validation."""

import pytest

import mindspore as ms
from mindspore import mint
from tests.mark_utils import arg_mark


@ms.jit
def pad_sequence_func(input_sequence):
    """Call mint.pad_sequence in Graph mode."""
    return mint.pad_sequence(input_sequence)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard')
def test_mint_pad_sequence_dynamic_scalar_input():
    """
    Feature: mint.pad_sequence input validation.
    Description: Pass a dynamic-length sequence whose elements are scalar tensors.
    Expectation: ValueError is raised instead of accessing invalid shape iterators.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    sequences = ms.mutable((ms.Tensor(1.0, ms.float32), ms.Tensor(2.0, ms.float32)), dynamic_len=True)

    with pytest.raises(ValueError, match="every sequence tensor must have rank >= 1"):
        pad_sequence_func(sequences)
