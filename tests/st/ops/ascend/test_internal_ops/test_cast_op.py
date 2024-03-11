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

import random
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.common.dtype as mstype

class CastNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cast = ops.Cast()

    def construct(self, input_x, dst_dtype):
        out = self.cast(input_x, dst_dtype)
        return out


def random_int_list(start, stop, length):
    """
    Feature: random_int_list
    Description: random_int_list
    Expectation: start <= out <= stop
    """
    random_list = []
    for _ in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def test_cast_net():
    """
    Feature: cast test case
    Description: cast test case
    Expectation: the result is correct
    """
    net = CastNet()
    size = 10

    # np_data = np.random.randn(size).astype(np.float32) * 10
    np_data = np.arange(size).astype(np.float32)
    np_data_bool = random_int_list(0, 1, size)

    print("test size : [", size, "]")

    bool_tensor = Tensor(np_data_bool, mstype.bool_)
    fp32_tensor = Tensor(np_data_bool, mstype.float32)
    fp16_tensor = Tensor(np_data_bool, mstype.float16)
    bf16_tensor = Tensor(np_data_bool, mstype.bfloat16)

    fp32_out = net(bool_tensor, mstype.float32)
    assert fp32_out.dtype == fp32_tensor.dtype
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("bool to fp32 success.")

    fp16_out = net(bool_tensor, mstype.float16)
    assert fp16_out.dtype == fp16_tensor.dtype
    assert np.allclose(fp16_out.asnumpy(), fp16_tensor.asnumpy(), 0.01, 0.01)
    print("bool to fp16 success.")

    bf16_out = net(bool_tensor, mstype.bfloat16)
    assert bf16_out.dtype == bf16_tensor.dtype
    fp32_out = net(bf16_out, mstype.float32)
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("bool to bf16 to fp32 success.")


    int8_tensor = Tensor(np_data, mstype.int8)
    int32_tensor = Tensor(np_data, mstype.int32)
    int64_tensor = Tensor(np_data, mstype.int64)

    fp32_tensor = Tensor(np_data, mstype.float32)
    fp16_tensor = Tensor(np_data, mstype.float16)
    bf16_tensor = Tensor(np_data, mstype.bfloat16)

    int64_out = net(int32_tensor, mstype.int64)
    assert int64_out.dtype == int64_tensor.dtype
    assert np.allclose(int64_out.asnumpy(), int64_tensor.asnumpy(), 0.01, 0.01)
    print("int32 to int64 success.")

    int32_out = net(int64_tensor, mstype.int32)
    assert int32_out.dtype == int32_tensor.dtype
    assert np.allclose(int32_out.asnumpy(), int32_tensor.asnumpy(), 0.01, 0.01)
    print("int64 to int32 success.")

    fp16_out = net(fp32_tensor, mstype.float16)
    assert fp16_out.dtype == fp16_tensor.dtype
    assert np.allclose(fp16_out.asnumpy(), fp16_tensor.asnumpy(), 0.01, 0.01)
    print("fp32 to fp16 success.")

    fp32_out = net(fp16_tensor, mstype.float32)
    assert fp32_out.dtype == fp32_tensor.dtype
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("fp16 to fp32 success.")

    bf16_out = net(fp32_tensor, mstype.bfloat16)
    assert bf16_out.dtype == bf16_tensor.dtype
    fp32_out = net(bf16_out, mstype.float32)
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("fp32 to bf16 to fp32 success.")

    fp32_out = net(bf16_tensor, mstype.float32)
    assert fp32_out.dtype == fp32_tensor.dtype
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("bf16 to fp32 success.")


    bf16_out = net(fp16_tensor, mstype.bfloat16)
    assert bf16_out.dtype == bf16_tensor.dtype
    fp32_out = net(bf16_out, mstype.float32)
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("fp16 to bf16 to fp32 success.")

    fp16_out = net(bf16_tensor, mstype.float16)
    assert fp16_out.dtype == fp16_tensor.dtype
    assert np.allclose(fp16_out.asnumpy(), fp16_tensor.asnumpy(), 0.01, 0.01)
    print("bf16 to fp16 success.")

    fp32_tensor = Tensor(np_data.astype(np.int8), mstype.float32)
    fp16_tensor = Tensor(np_data.astype(np.int8), mstype.float16)
    bf16_tensor = Tensor(np_data.astype(np.int8), mstype.bfloat16)

    fp32_out = net(int8_tensor, mstype.float32)
    assert fp32_out.dtype == fp32_tensor.dtype
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("int8 to fp32 success.")

    fp16_out = net(int8_tensor, mstype.float16)
    assert fp16_out.dtype == fp16_tensor.dtype
    assert np.allclose(fp16_out.asnumpy(), fp16_tensor.asnumpy(), 0.01, 0.01)
    print("int8 to fp16 success.")

    bf16_out = net(int8_tensor, mstype.bfloat16)
    assert bf16_out.dtype == bf16_tensor.dtype
    fp32_out = net(bf16_out, mstype.float32)
    assert np.allclose(fp32_out.asnumpy(), fp32_tensor.asnumpy(), 0.01, 0.01)
    print("int8 to bf16 to fp32 success.")
    