#Copyright 2024 Huawei Technologies Co., Ltd
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
"""
Test Cast plugin custom ops.
"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, Parameter
import mindspore .common.dtype as mstype
from mindspore.ops import Cast
from bfloat16 import bfloat16

class CastNet(nn.Cell):
    """
    CastNet.
    """
    def __init__(self):
        super().__init__()
        self.cast = Cast()
#>>> type_dst = mindspore.int32
#>>> cast = ops.Cast()
#>>> output = cast(input_x, type_dst)
#>>> print(output.dtype)

    def construct(self, input_x, dst_dtype):
        out = self.cast(input_x, dst_dtype)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('size', [10, 128])
def test_cast_net_int(size):
    """
    Feature: Test Cast.
    Description: Test Cast.
    Expectation: Assert that results are consistent with numpy.
    """
    context.set_context(device_target="Ascend")
    #os.environ['GRAPH_OP_RUN'] = '1'
    net = CastNet()

    np_data = np.arange(size).astype(np.float32)
    int32_tensor = Tensor(np_data, mstype.int32)
    int64_tensor = Tensor(np_data, mstype.int64)

    int64_out = net(int32_tensor, mstype.int64)
    assert int64_out.dtype == int64_tensor.dtype
    assert np.allclose(int64_out.asnumpy(), int64_tensor.asnumpy())

    int32_out = net(int64_tensor, mstype.int32)
    assert int32_out.dtype == int32_tensor.dtype
    assert np.allclose(int32_out.asnumpy(), int32_tensor.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('size', [10, 128])
def test_cast_net_float(size):
    """
    Feature: Test Cast.
    Description: Test Cast.
    Expectation: Assert that results are consistent with numpy.
    """
    context.set_context(device_target="Ascend")
    #os.environ['GRAPH_OP_RUN'] = '1'
    net = CastNet()

    np_data = np.arange(size).astype(np.float32)

    fp32_np_data = np_data.astype(np.float32)
    fp16_np_data = np_data.astype(np.float16)
    bf16_np_data = np_data.astype(bfloat16)

    fp32_ms_tensor = Tensor(np_data, mstype.float32)
    fp16_ms_tensor = Tensor(np_data, mstype.float16)
    bf16_ms_tensor = Tensor(np_data, mstype.bfloat16)

    #
    # fp32 and bf16 cast
    #
    bf16_out = net(fp32_ms_tensor, mstype.bfloat16)
    assert bf16_out.dtype == bf16_ms_tensor.dtype
    assert np.allclose(bf16_out.asnumpy(), bf16_np_data)

    fp32_out = net(bf16_ms_tensor, mstype.float16)
    assert fp32_out.dtype == fp32_ms_tensor.dtype
    assert np.allclose(fp32_out.asnumpy(), fp32_np_data)


    #
    # fp16 and bf16 cast
    #
    bf16_out = net(fp16_ms_tensor, mstype.bfloat16)
    assert bf16_out.dtype == bf16_ms_tensor.dtype
    assert np.allclose(bf16_out.asnumpy(), bf16_np_data)

    fp16_out = net(bf16_ms_tensor, mstype.float16)
    assert fp16_out.dtype == fp16_ms_tensor.dtype
    assert np.allclose(fp16_out.asnumpy(), fp16_np_data)
