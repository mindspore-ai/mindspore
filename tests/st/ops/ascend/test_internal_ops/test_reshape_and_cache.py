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
"""
Test ReshapeAndCache plugin custom ops.
"""
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context, Parameter
from mindspore.ops import ReshapeAndCache

num_slots = 100
slot_size = 128
b = 26
s = 32
h = 40
d = 128

class ReshapeAndCacheNet(nn.Cell):
    """
    ReshapeAndCacheNet.
    """
    def __init__(self):
        super().__init__()
        self.reshape_and_cache = ReshapeAndCache()

    def construct(self, key, value, key_cache, value_cache, slot_map):
        out = self.reshape_and_cache(key, value, key_cache, value_cache, slot_map)
        return out


def np_inference(key, value, key_cache, value_cache, slot_map):
    """
    np_inference
    """
    key_cache_ans = key_cache.copy()
    value_cache_ans = value_cache.copy()
    for i, slot in enumerate(slot_map):
        slot_idx = slot // key_cache.shape[1]
        slot_offset = slot % key_cache.shape[1]
        key_cache_ans[slot_idx][slot_offset] = key[i]
        value_cache_ans[slot_idx][slot_offset] = value[i]
    return key_cache_ans, value_cache_ans


def create_ms_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map):
    """
    create inputs
    """
    ms_key = Tensor(np_k)
    ms_value = Tensor(np_v)
    ms_key_cache = Parameter(Tensor(np_k_cache))
    ms_value_cache = Parameter(Tensor(np_v_cache))
    ms_slot_map = Tensor(np_slot_map)
    return ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map


def create_np_inputs():
    """
    create_np_inputs
    """
    cache_shape = (num_slots, slot_size, h, d)
    update_shape = (b * s, h, d)
    key_update = np.random.rand(*update_shape).astype(np.float16)
    value_update = np.random.rand(*update_shape).astype(np.float16)
    key_cache = np.random.rand(*cache_shape).astype(np.float16)
    value_cache = np.random.rand(*cache_shape).astype(np.float16)

    num_tokens = update_shape[0]
    slot_map = np.random.choice(np.arange(num_tokens), num_tokens,
                                replace=False).astype(np.int32)

    return key_update, value_update, key_cache, value_cache, slot_map


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_reshape_and_cache_net():
    """
    Feature: Test ReshapeAndCache.
    Description: Test float16 inputs.
    Expectation: Assert that results are consistent with numpy.
    """
    context.set_context(device_target="Ascend")
    net = ReshapeAndCacheNet()
    np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_np_inputs()
    np_k_cache_out, np_v_cache_out = np_inference(np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
    _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map)
    assert np.allclose(ms_k_cache.asnumpy(), np_k_cache_out, 0.001, 0.001)
    assert np.allclose(ms_v_cache.asnumpy(), np_v_cache_out, 0.001, 0.001)
