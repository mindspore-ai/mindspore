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


class PadV3GePassNet(nn.Cell):
    """
    PadV3GePassNet.
    """

    def __init__(self):
        super().__init__()
        self.pad = ops.PadV3()
        self.concat = ops.Concat(axis=0)
        self.sub = ops.Sub()
        self.up_to_len = 10

    def construct(self, key):
        pad_length = (
            self.sub(ms.Tensor(self.up_to_len, mstype.int32), ops.dyn_shape(key)[-2])
            .reshape((1,))
            .astype(mstype.int32)
        )
        key_paddings = self.concat(
            (
                Tensor([0, 0, 0], mstype.int32),
                pad_length,
                Tensor([0, 0, 0, 0], mstype.int32),
            )
        )
        key_present = self.pad(key, key_paddings, Tensor(0, mstype.float16))
        return key_present


def dummy_tensor(shape, dtype):
    """create dummy tensor"""
    if None in shape:
        return Tensor(shape=shape, dtype=dtype)
    return Tensor(np.ones(shape=tuple(shape)), dtype=dtype)


def export_model():
    """
    export model
    """
    in_key = dummy_tensor(shape=[1, 2, None, 2], dtype=ms.float16)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = PadV3GePassNet()
    file_name = "padv3_ge_pass"
    ms.export(net, in_key, file_name=file_name, file_format="MINDIR")
    model_name = file_name + ".mindir"
    assert os.path.exists(model_name)
    return model_name


def inference_common():
    """
    def inference_padv3_ge_pass
    """
    time_start_total = time.time()
    model_path = export_model()

    lite_ctx0 = mslite.Context()
    lite_ctx0.target = ["ascend"]
    lite_ctx0.ascend.device_id = 0
    lite_ctx0.ascend.provider = "ge"
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, lite_ctx0)

    # input data
    np_data = np.zeros((1, 2, 4, 2), np.float16)
    # change the first 2 elements of the third dimension to 1
    np_data[:, :, :2, :] = 1.0
    outputs = model.predict([mslite.Tensor(np_data)])
    result = np.zeros((1, 2, 10, 2), np.float16)
    # change the first 2 elements of the third dimension to 1
    result[:, :, :2, :] = 1.0
    os.remove(model_path)
    print(f"predict cost total {(time.time() - time_start_total)*1000} ms", flush=True)
    print("gold shape:")
    print(result.shape)
    print(result)
    print("model output shape:")
    np_out = outputs[0].get_data_to_numpy()
    print(np_out.shape)
    print(np_out)

    assert np_out.shape == result.shape
    assert (np_out == result).all()


if __name__ == "__main__":
    print("test_padv3_ge_pass.py: begin run testcases.")
    backend = sys.argv[1]
    if backend == "Ascend":
        inference_common()
    else:
        print(f"test_padv3_ge_pass.py: skip backend {backend}!")
    print("test_padv3_ge_pass.py: run testcases success.")
