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
"""
Test DecoderKVCache plugin custom ops.
"""
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.ops.operations._inner_ops import DecoderKVCache
from tests.mark_utils import arg_mark


b = 26
h = 40
s = 32
d = 128
us = 1
ps = s - us


class DecoderKVCacheNet(nn.Cell):
    """
    DecoderKVCacheNet.
    """

    def __init__(self):
        super().__init__()
        self.add = ops.Add()
        self.sub = ops.Sub()
        self.decoder_k_v_cache = DecoderKVCache()
        self.seq_len_axis = Tensor(np.array([2, 0, 0, 0]).astype(np.int64))
        self.depend = ops.Depend()

    def construct(self, cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len):
        out = self.decoder_k_v_cache(cache, update, valid_seq_len, batch_index, self.seq_len_axis,
                                     new_max_seq_len, cur_max_seq_len)
        cache = self.depend(cache, out)
        add_out = self.add(cache, 1)
        sub_out = self.sub(add_out, 1)
        return sub_out


def np_inference(cache, update, valid_seq_len):
    """
    np_inference
    """
    ans = cache.copy()
    for b_idx in range(cache.shape[0]):
        s_idx = valid_seq_len[b_idx]
        if s_idx < 0:
            continue
        ans[b_idx, :, s_idx, :] = update[b_idx, :, 0, :]
    return ans


def create_ms_inputs():
    """
    create inputs
    """
    cache = np.random.rand(b, h, s, d).astype(np.float16)
    update = np.random.rand(b, h, us, d).astype(np.float16)
    valid_seq_len = np.random.randint(-1, s, size=b).astype(np.int64)
    batch_index = np.array([1]).astype(np.int64)
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


def create_np_inputs(cache, update, valid_seq_len):
    """
    create_np_inputs
    """
    return cache.asnumpy(), update.asnumpy(), valid_seq_len.asnumpy()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_decoder_k_v_cache_net():
    """
    Feature: Test DecoderKVCache.
    Description: Test float16 inputs.
    Expectation: Assert that results are consistent with numpy.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = DecoderKVCacheNet()
    cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len = create_ms_inputs()
    output = net(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)
    np_cache, np_update, np_valid_seq_len = create_np_inputs(cache, update, valid_seq_len)
    expect_output = np_inference(np_cache, np_update, np_valid_seq_len)
    assert np.allclose(output.asnumpy(), expect_output, 0.001, 0.001)
