# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
"""
Test PromptKVCache plugin custom ops.
"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, Parameter
import mindspore.common.dtype as mstype
from mindspore.ops.operations._inner_ops import PromptKVCache
from tests.st.utils import test_utils

b = 4
h = 4
s = 4096
d = 32
ub = 1
us = s
ps = s - us


class PromptKVCacheNet(nn.Cell):
    """
    PromptKVCacheNet.
    """

    def __init__(self, padding_mode):
        super().__init__()
        self.sub = ops.Sub()
        self.add = ops.Add()
        self.concat_dim2 = ops.Concat(axis=2)
        self.prompt_k_v_cache = PromptKVCache(padding_mode)
        self.pad_update_zero_tensor = Parameter(ops.zeros((ub, h, ps, d), mstype.float16))
        self.depend = ops.Depend()

    def construct(self, cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len):
        update_pad = self.concat_dim2((update, self.pad_update_zero_tensor))
        out = self.prompt_k_v_cache(cache, update_pad, valid_seq_len, batch_index, seq_len_axis,
                                    new_max_seq_len, cur_max_seq_len)
        cache = self.depend(cache, out)
        add_out = self.add(cache, 1)
        sub_out = self.sub(add_out, 1)
        return sub_out


def np_inference(cache, update, batch_index):
    """
    np_inference
    """
    zeros_ans = np.zeros(cache.shape, cache.dtype)
    update_s = update.shape[2]
    for idx in batch_index:
        if idx < 0:
            continue
        cache[idx, :] = zeros_ans[idx, :]
        cache[idx, :, 0:update_s, :] = update
    return cache


def create_ms_inputs():
    """
    create inputs
    """
    cache_shape = (b, h, s, d)
    update_shape = (ub, h, us, d)
    cache = np.random.rand(*cache_shape).astype(np.float16)
    update = np.random.rand(*update_shape).astype(np.float16)
    valid_seq_len = np.array([0, 1, 2, 3]).astype(np.int64)
    batch_index = np.random.randint(-1, b, size=ub).astype(np.int64)
    seq_len_axis = np.array([2]).astype(np.int64)
    new_max_seq_len = np.array([s]).astype(np.int64)
    cur_max_seq_len = np.array([s]).astype(np.int64)

    ms_cache = Tensor(cache)
    ms_update = Tensor(update)
    ms_valid_seq_len = Tensor(valid_seq_len)
    ms_batch_index = Tensor(batch_index)
    ms_seq_len_axis = Tensor(seq_len_axis)
    ms_new_max_seq_len = Tensor(new_max_seq_len)
    ms_cur_max_seq_len = Tensor(cur_max_seq_len)
    return (ms_cache, ms_update, ms_valid_seq_len, ms_batch_index, ms_seq_len_axis,
            ms_new_max_seq_len, ms_cur_max_seq_len)


def create_np_inputs(cache, update, batch_index):
    """
    create_np_inputs
    """
    return cache.asnumpy(), update.asnumpy(), batch_index.asnumpy()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@test_utils.run_test_with_On
def test_prompt_k_v_cache_net():
    """
    Feature: Test PromptKVCache.
    Description: Test float16 inputs.
    Expectation: Assert that results are consistent with numpy.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = PromptKVCacheNet("right")
    cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len = create_ms_inputs()
    output = net(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)
    np_cache, np_update, batch_index = create_np_inputs(cache, update, batch_index)
    expect_output = np_inference(np_cache, np_update, batch_index)
    assert np.allclose(output.asnumpy(), expect_output, 0.001, 0.001)
