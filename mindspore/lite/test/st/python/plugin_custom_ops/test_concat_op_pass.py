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
Test concat op pass.
"""
import os
import sys
import time
import numpy as np
import mindspore_lite as mslite
import mindspore as ms
from mindspore import Tensor, ops, nn
import mindspore.common.dtype as mstype
from mindspore import context


class ConcatOpPassNet(nn.Cell):
    """
    KVCacheMgrNet.
    """
    def __init__(self):
        super().__init__()
        self.pad = ops.PadV3()
        self.concat = ops.Concat(axis=0)

    def construct(self, key):
        pad_length = key.astype(mstype.int64)
        key_paddings = self.concat((Tensor([0, 0, 0, 0, 0], mstype.int64), pad_length, Tensor([0, 0], mstype.int64)))
        return key_paddings

def dummy_tensor(shape, dtype):
    """create dummy tensor"""
    if None in shape:
        return Tensor(shape=shape, dtype=dtype)
    return Tensor(np.ones(shape=tuple(shape)), dtype=dtype)

def export_model():
    """
    export model
    """
    in_key = dummy_tensor(shape=[None], dtype=ms.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = ConcatOpPassNet()
    file_name = "concat_op_pass"
    ms.export(net, in_key, file_name=file_name, file_format="MINDIR")
    model_name = file_name + ".mindir"
    assert os.path.exists(model_name)
    return model_name


def inference():
    """
    def inference_concat_op_pass
    """
    time_start_total = time.time()
    model_path = export_model()

    lite_ctx0 = mslite.Context()
    lite_ctx0.target = ["ascend"]
    lite_ctx0.ascend.device_id = 0
    lite_ctx0.ascend.provider = "ge"
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, lite_ctx0)
    # warm up
    np_data = np.ones((1), np.int64)
    outputs = model.predict([mslite.Tensor(np_data)])
    result = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    os.remove(model_path)
    print(f"predict cost total {(time.time() - time_start_total)*1000} ms", flush=True)
    assert (outputs[0].get_data_to_numpy() == result).all()


if __name__ == '__main__':
    print("test_concat_op_pass.py: begin run testcases.")
    backend = sys.argv[1]
    if backend == "Ascend":
        inference()
    else:
        print(f'test_concat_op_pass.py: skip backend {backend}!')
    print("test_concat_op_pass.py: run testcases success.")
