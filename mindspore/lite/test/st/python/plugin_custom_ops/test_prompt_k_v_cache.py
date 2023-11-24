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
Test PromptKVCache plugin custom ops.
"""
import os
import numpy as np
import mindspore_lite as mslite
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train.serialization import export
from mindspore.ops.operations._inner_ops import PromptKVCache

b = 40
h = 4
s = 1024
d = 32
ub = 40
us = 512
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

    def construct(self, cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len):
        out = self.prompt_k_v_cache(cache, update, valid_seq_len, batch_index, seq_len_axis,
                                    new_max_seq_len, cur_max_seq_len)
        add_out = self.add(cache, 1)
        sub_out = self.sub(add_out, 1)
        return sub_out


def np_inference(cache, update, valid_seq_len, batch_index):
    """
    np_inference
    """
    s_ = cache.shape[2]
    us_ = update.shape[2]
    for i in range(batch_index.size):
        b_idx = batch_index[i]
        s_idx = valid_seq_len[i]
        if b_idx < 0:
            continue
        if s_idx < 0 or s_idx + us > s_:
            continue
        cache[b_idx, :, s_idx:s_idx + us_, :] = update[i]

    return cache


def create_numpy_inputs():
    """
    create inputs
    """
    cache_shape = (b, h, s, d)
    update_shape = (ub, h, us, d)
    cache = np.random.rand(*cache_shape).astype(np.float16)
    update = np.random.rand(*update_shape).astype(np.float16)
    valid_seq_len = np.random.randint(-1, s, size=ub).astype(np.int64)
    batch_index = np.random.choice(np.arange(-1, b), size=ub, replace=False).astype(np.int64)
    seq_len_axis = np.array([2]).astype(np.int64)
    new_max_seq_len = np.array([s]).astype(np.int64)
    cur_max_seq_len = np.array([s]).astype(np.int64)
    return (cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)


def create_ms_inputs():
    """
    create inputs
    """
    cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len = create_numpy_inputs()
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
    return (cache, update, batch_index)


def create_lite_inputs(cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len):
    """
    create_lite_inputs
    """
    cache = mslite.Tensor(cache)
    update = mslite.Tensor(update)
    valid_seq_len = mslite.Tensor(valid_seq_len)
    batch_index = mslite.Tensor(batch_index)
    seq_len_axis = mslite.Tensor(seq_len_axis)
    new_max_seq_len = mslite.Tensor(new_max_seq_len)
    cur_max_seq_len = mslite.Tensor(cur_max_seq_len)
    return (cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len)


def inference_prompt_k_v_cache(mindir_model):
    """
    inference model
    """

    lite_ctx1 = mslite.Context()
    lite_ctx1.target = ["ascend"]
    lite_ctx1.ascend.device_id = 0
    lite_ctx1.ascend.provider = "ge"

    model = mslite.Model()
    model.build_from_file(
        mindir_model, mslite.ModelType.MINDIR, lite_ctx1, "", {})

    for i in range(100):
        cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, \
            cur_max_seq_len = create_numpy_inputs()
        input_lists = list(create_lite_inputs(cache, update, valid_seq_len, batch_index, seq_len_axis,
                                              new_max_seq_len, cur_max_seq_len))
        mslite_output = model.predict(input_lists)
        np_cache, np_update, batch_index = create_np_inputs(
            cache, update, batch_index)
        expect_output = np_inference(np_cache, np_update, valid_seq_len, batch_index)
        assert np.allclose(
            mslite_output[0].get_data_to_numpy(), expect_output, 0.001, 0.001)
        print(f"prompt_k_v_cache st {i} times: inference success.")


def export_prompt_k_v_cache_model():
    """
    export model
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len = create_ms_inputs()

    net = PromptKVCacheNet("right")
    file_name = "prompt_k_v_cache_primitive"

    export(net, cache, update, valid_seq_len, batch_index, seq_len_axis, new_max_seq_len, cur_max_seq_len,
           file_name=file_name, file_format='MINDIR')
    model_name = file_name + ".mindir"
    assert os.path.exists(model_name)
    return model_name


if __name__ == '__main__':
    model_path = export_prompt_k_v_cache_model()
    print("prompt_k_v_cache st : export success path: ", model_path)

    inference_prompt_k_v_cache(model_path)
    print(f"prompt_k_v_cache st : inference end.")
