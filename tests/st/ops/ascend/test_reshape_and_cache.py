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
        self.reshape_and_cache(key, value, key_cache, value_cache, slot_map)
        return key_cache, value_cache


def np_inference(key, value, key_cache, value_cache, slot_map):
    """
    np_inference
    """
    slot_size_val = key_cache.shape[1]
    s_val = key.shape[1]
    for i, slot in enumerate(slot_map):
        slot_idx = slot // slot_size_val
        slot_offset = slot % slot_size_val
        b_idx = i // s_vals
        s_idx = i % s_val
        key_token = key[b_idx][s_idx]
        value_token = value[b_idx][s_idx]
        key_cache[slot_idx][slot_offset] = key_token
        value_cache[slot_idx][slot_offset] = value_token
    return key_cache, value_cache


def create_ms_inputs():
    """
    create inputs
    """
    cache_shape = (num_slots, slot_size, h * d)
    update_shape = (b, s, h * d)
    key_cache = np.random.rand(*cache_shape).astype(np.float16)
    key_update = np.random.rand(*update_shape).astype(np.float16)
    value_cache = np.random.rand(*cache_shape).astype(np.float16)
    value_update = np.random.rand(*update_shape).astype(np.float16)

    ms_key = Tensor(key_update)
    ms_key_cache = Parameter(Tensor(key_cache))
    ms_value = Tensor(value_update)
    ms_value_cache = Parameter(Tensor(value_cache))

    num_tokens = b * s
    slot_map = np.random.choice(np.arange(num_tokens), num_tokens,
                                replace=False).astype(np.int32)
    ms_slot_map = Tensor(slot_map)
    return (ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map)


def create_np_inputs(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map):
    """
    create_np_inputs
    """
    return ms_k.asnumpy(), ms_v.asnumpy(), ms_k_cache.asnumpy(),\
           ms_v_cache.asnumpy(), ms_slot_map.asnumpy()


@pytest.mark.level2
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
    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs()
    ms_cache_k, ms_cache_v = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map)
    k, v, k_cache, v_cache, slot_map = create_np_inputs(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map)
    np_cache_k, np_cache_v = np_inference(k, v, k_cache, v_cache, slot_map)
    assert np.allclose(ms_cache_k.asnumpy(), np_cache_k, 0.001, 0.001)
    assert np.allclose(ms_cache_v.asnumpy(), np_cache_v, 0.001, 0.001)
