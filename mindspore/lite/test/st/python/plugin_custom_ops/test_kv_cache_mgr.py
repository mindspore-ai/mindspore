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
Test KVCacheMgr plugin custom ops.
"""
import os
import sys
import time
import numpy as np
import mindspore_lite as mslite
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.train.serialization import export
from mindspore import Tensor
from mindspore import context


class KVCacheMgrNet(nn.Cell):
    """
    KVCacheMgrNet.
    """
    def __init__(self, batch_size, src_seq_length):
        super().__init__()
        self.mul = P.Mul()
        self.add = P.Add()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.dtype = ms.float16

        seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
        self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), ms.int32)
        self.equal = P.Equal()
        self.sub = P.Sub()

    def construct(self, key_past, key, value_past, value, batch_valid_length):
        current_index = F.reshape(batch_valid_length, (-1, 1, 1))
        current_mask = F.cast(self.equal(self.range, current_index), self.dtype)
        # Pad the key and value to seq_length with only the position index not zero
        current_key = self.mul(key, self.expand_dims(current_mask, 3))
        current_value = self.mul(value, self.expand_dims(current_mask, 3))
        # Concat the previous saved state and current state
        key = self.add(key_past, current_key)
        value = self.add(value_past, current_value)

        ans = self.sub(key, value)
        return ans


def create_shapes():
    batch_size = 1
    num_head = 40
    seq_length = 1024
    update_seq_length = 1
    size_pre_head = 128
    past_shape = (batch_size, num_head, seq_length, size_pre_head)
    cur_shape = (batch_size, num_head, update_seq_length, size_pre_head)
    return past_shape, cur_shape


def create_inputs():
    """
    create inputs.
    """
    past_shape, cur_shape = create_shapes()

    key_past = Tensor(np.random.rand(*past_shape), ms.float16)
    key_cur = Tensor(np.random.rand(*cur_shape), ms.float16)
    value_past = Tensor(np.random.rand(*past_shape), ms.float16)
    value_cur = Tensor(np.random.rand(*cur_shape), ms.float16)
    index = Tensor(shape=(1,), dtype=ms.int32, init=1)
    return (key_past, key_cur, value_past, value_cur, index)


def create_lite_inputs():
    """
    create lite inputs.
    """
    past_shape, cur_shape = create_shapes()

    key_past = mslite.Tensor(np.zeros(past_shape, np.float16))
    key_cur = mslite.Tensor(np.random.rand(*cur_shape).astype(np.float16))
    value_past = mslite.Tensor(np.zeros(past_shape, np.float16))
    value_cur = mslite.Tensor(np.random.rand(*cur_shape).astype(np.float16))
    index = mslite.Tensor(np.ones(1).astype(np.int32))
    return (key_past, key_cur, value_past, value_cur, index)


def export_model():
    """
    export model
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    key_past, key_cur, value_past, value_cur, index = create_inputs()
    batch_size = key_past.shape[0]
    src_seq_length = key_past.shape[-2]

    net = KVCacheMgrNet(batch_size, src_seq_length)
    file_name = "kv_cache_mgr_net"

    export(net, key_past, key_cur, value_past, value_cur,
           index, file_name=file_name, file_format='MINDIR')
    model_name = file_name + ".mindir"
    assert os.path.exists(model_name)
    return model_name


def inference_kv_cache_mgr():
    """
    def inference_kv_cache_mgr
    """
    time_start_total = time.time()
    model_path = export_model()
    input_lists = list(create_lite_inputs())

    lite_ctx0 = mslite.Context()
    lite_ctx0.target = ["ascend"]
    lite_ctx0.ascend.device_id = 0
    lite_ctx0.ascend.provider = "ge"
    model0 = mslite.Model()
    model0.build_from_file(model_path, mslite.ModelType.MINDIR, lite_ctx0)
    # warm up
    outputs0 = model0.predict(input_lists)
    time_start = time.time()
    outputs0 = model0.predict(input_lists)
    print(f"predict plugin_custom_ops=None cost {(time.time() - time_start)*1000} ms", flush=True)

    lite_ctx1 = mslite.Context()
    lite_ctx1.target = ["ascend"]
    lite_ctx1.ascend.device_id = 0
    lite_ctx1.ascend.provider = "ge"
    dict1 = {"ascend_context": {"plugin_custom_ops": "All"}}
    model1 = mslite.Model()
    model1.build_from_file(model_path, mslite.ModelType.MINDIR, lite_ctx1, "", dict1)
    # warm up
    outputs1 = model1.predict(input_lists)
    time_start = time.time()
    outputs1 = model1.predict(input_lists)
    print(f"predict plugin_custom_ops=All cost {(time.time() - time_start)*1000} ms", flush=True)

    os.remove(model_path)
    print(f"predict cost total {(time.time() - time_start_total)*1000} ms", flush=True)
    assert (outputs0[0].get_data_to_numpy() == outputs1[0].get_data_to_numpy()).all()


if __name__ == '__main__':
    print("test_kv_cache_mgr_plugin_custom_ops.py: begin run testcases.")
    backend = sys.argv[1]
    if backend == "Ascend":
        inference_kv_cache_mgr()
    else:
        print(f'test_kv_cache_mgr_plugin_custom_ops.py: skip backend {backend}!')
    print("test_kv_cache_mgr_plugin_custom_ops.py: run testcases success.")
