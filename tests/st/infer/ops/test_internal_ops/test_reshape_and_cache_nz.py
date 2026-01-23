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
"""
Test ReshapeAndCacheNz ops.
"""
import numpy as np
import pytest
from mindspore import Tensor, context, Parameter, ops, nn
from tests.mark_utils import arg_mark


num_slots = 512
b = 1
s = 5
h = 1
d = 576


class ReshapeAndCacheAll(nn.Cell):
    """
    ReshapeAndCacheNet.
    """

    def __init__(self):
        super().__init__()
        self.reshape_and_cache = ops.ReshapeAndCache()

    def construct(self, key, key_cache, slot_map):
        out = self.reshape_and_cache(key, None, key_cache, None, slot_map)
        return out


def np_inference(slot_size, key, value, key_cache, value_cache, slot_map):
    """
    np_inference
    """
    key_tmp = key.copy()
    value_tmp = value.copy()
    key_cache_ans = key_cache.copy()
    value_cache_ans = value_cache.copy()
    head = h
    head_dim = d
    key_tmp = key_tmp.reshape(-1, head, head_dim)
    value_tmp = value_tmp.reshape(-1, head, head_dim)
    key_cache_ans = key_cache_ans.reshape(num_slots, slot_size, head, head_dim)
    value_cache_ans = value_cache_ans.reshape(
        num_slots, slot_size, head, head_dim)
    for i, slot in enumerate(slot_map):
        slot_idx = slot // key_cache.shape[1]
        slot_offset = slot % key_cache.shape[1]
        key_cache_ans[slot_idx][slot_offset] = key_tmp[i]
        value_cache_ans[slot_idx][slot_offset] = value_tmp[i]
    key_cache_ans = key_cache_ans.reshape(
        num_slots, slot_size, head * head_dim)
    value_cache_ans = value_cache_ans.reshape(
        num_slots, slot_size, head * head_dim)
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


def create_np_inputs(slot_size, dtype=np.float16, kv_dim=3):
    """
    create_np_inputs
    """
    cache_shape = (num_slots, slot_size, h * d)
    if kv_dim == 2:
        update_shape = (b * s, h * d)
        num_tokens = update_shape[0]
    elif kv_dim == 3:
        update_shape = (b, s, h * d)
        num_tokens = update_shape[0] * update_shape[1]
    else:
        raise Exception(
            "Key's dim should be 2 or 3, but got {0}".format(kv_dim))
    key_update = np.random.rand(*update_shape).astype(dtype)
    value_update = np.random.rand(*update_shape).astype(dtype)
    key_cache = np.random.rand(*cache_shape).astype(dtype)
    value_cache = np.random.rand(*cache_shape).astype(dtype)

    slot_map = np.random.choice(np.arange(num_tokens), num_tokens,
                                replace=False).astype(np.int32)
    return key_update, value_update, key_cache, value_cache, slot_map


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=172.00)
@pytest.mark.parametrize('slot_size', [16, 128])
@pytest.mark.parametrize('kv_dim', [2, 3])
def test_reshape_and_cache_nz_swft(slot_size, kv_dim):
    """
    Feature: Test ReshapeAndCacheNz with swft kernels.
    Description: Test float16 inputs.
    Expectation: Assert that results are consistent with numpy.
    """
    context.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = ReshapeAndCacheAll()

    np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_np_inputs(
        slot_size, np.float16, kv_dim)
    np_k_cache_out, _ = np_inference(
        slot_size, np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

    ms_k, _, ms_k_cache, _, ms_slot_map = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
    _ = net(ms_k, ms_k_cache, ms_slot_map)
    assert np.allclose(ms_k_cache.asnumpy(), np_k_cache_out, 0.001, 0.001)
