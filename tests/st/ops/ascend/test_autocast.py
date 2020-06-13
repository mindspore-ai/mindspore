# Copyright 2020 Huawei Technologies Co., Ltd
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
"""multitype_ops directory test case"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import functional as F
import mindspore.context as context


class TensorIntAutoCast(nn.Cell):
    def __init__(self,):
        super(TensorIntAutoCast, self).__init__()
        self.i = 2

    def construct(self, t):
        z = F.tensor_mul(t, self.i)
        return z


class TensorFPAutoCast(nn.Cell):
    def __init__(self,):
        super(TensorFPAutoCast, self).__init__()
        self.f = 1.2

    def construct(self, t):
        z = F.tensor_mul(t, self.f)
        return z


class TensorBoolAutoCast(nn.Cell):
    def __init__(self,):
        super(TensorBoolAutoCast, self).__init__()
        self.f = True

    def construct(self, t):
        z = F.tensor_mul(t, self.f)
        return z


class TensorAutoCast(nn.Cell):
    def __init__(self,):
        super(TensorAutoCast, self).__init__()

    def construct(self, t1, t2):
        z = F.tensor_mul(t1, t2)
        return z


def test_tensor_auto_cast():
    context.set_context(mode=context.GRAPH_MODE)
    Tensor([True, False], mstype.bool_)
    t_uint8 = Tensor(np.ones([2, 1, 2, 2]), mstype.uint8)
    t_int8 = Tensor(np.ones([2, 1, 2, 2]), mstype.int8)
    t_int16 = Tensor(np.ones([2, 1, 2, 2]), mstype.int16)
    t_int32 = Tensor(np.ones([2, 1, 2, 2]), mstype.int32)
    t_int64 = Tensor(np.ones([2, 1, 2, 2]), mstype.int64)
    t_fp16 = Tensor(np.ones([2, 1, 2, 2]), mstype.float16)
    t_fp32 = Tensor(np.ones([2, 1, 2, 2]), mstype.float32)
    t_fp64 = Tensor(np.ones([2, 1, 2, 2]), mstype.float64)
    net = TensorAutoCast()
    rs = net(t_uint8, t_int8)
    assert rs.dtype == mstype.int16
    rs = net(t_uint8, t_int16)
    assert rs.dtype == mstype.int16
    rs = net(t_uint8, t_int32)
    assert rs.dtype == mstype.int32
    rs = net(t_uint8, t_int64)
    assert rs.dtype == mstype.int64
    rs = net(t_int8, t_int16)
    assert rs.dtype == mstype.int16
    rs = net(t_int8, t_int32)
    assert rs.dtype == mstype.int32
    rs = net(t_int8, t_int64)
    assert rs.dtype == mstype.int64
    rs = net(t_int16, t_int32)
    assert rs.dtype == mstype.int32
    rs = net(t_int16, t_int64)
    assert rs.dtype == mstype.int64
    rs = net(t_int32, t_int64)
    assert rs.dtype == mstype.int64

    rs = net(t_fp16, t_fp32)
    assert rs.dtype == mstype.float32
    rs = net(t_fp16, t_fp64)
    assert rs.dtype == mstype.float64
    rs = net(t_fp32, t_fp64)
    assert rs.dtype == mstype.float64

    rs = net(t_uint8, t_fp16)
    assert rs.dtype == mstype.float16
    rs = net(t_uint8, t_fp32)
    assert rs.dtype == mstype.float32
    rs = net(t_uint8, t_fp64)
    assert rs.dtype == mstype.float64
    rs = net(t_int8, t_fp64)
    assert rs.dtype == mstype.float64
    rs = net(t_int16, t_fp64)
    assert rs.dtype == mstype.float64
    rs = net(t_int32, t_fp64)
    assert rs.dtype == mstype.float64
    rs = net(t_int64, t_fp64)
    assert rs.dtype == mstype.float64

    rs = net(t_fp16, t_int8)
    assert rs.dtype == mstype.float16
    rs = net(t_fp16, t_uint8)
    assert rs.dtype == mstype.float16
    rs = net(t_fp16, t_int16)
    assert rs.dtype == mstype.float16
    rs = net(t_fp16, t_int32)
    assert rs.dtype == mstype.float16
    rs = net(t_fp16, t_int64)
    assert rs.dtype == mstype.float16

    tint = TensorIntAutoCast()
    rs = tint(t_uint8)
    assert rs.dtype == mstype.uint8
    rs = tint(t_int8)
    assert rs.dtype == mstype.int8
    rs = tint(t_int16)
    assert rs.dtype == mstype.int16
    rs = tint(t_int32)
    assert rs.dtype == mstype.int32
    rs = tint(t_int64)
    assert rs.dtype == mstype.int64
    rs = tint(t_fp16)
    assert rs.dtype == mstype.float16
    rs = tint(t_fp32)
    assert rs.dtype == mstype.float32
    rs = tint(t_fp64)
    assert rs.dtype == mstype.float64
    tfp = TensorFPAutoCast()
    rs = tfp(t_uint8)
    assert rs.dtype == mstype.float32
    rs = tfp(t_int8)
    assert rs.dtype == mstype.float32
    rs = tfp(t_int16)
    assert rs.dtype == mstype.float32
    rs = tfp(t_int32)
    assert rs.dtype == mstype.float32
    rs = tfp(t_int64)
    assert rs.dtype == mstype.float32
    rs = tfp(t_fp16)
    assert rs.dtype == mstype.float32
    rs = tfp(t_fp32)
    assert rs.dtype == mstype.float32
    rs = tfp(t_fp64)
    assert rs.dtype == mstype.float64

    t_uint16 = Tensor(np.ones([2, 1, 2, 2]), mstype.uint16)
    t_uint32 = Tensor(np.ones([2, 1, 2, 2]), mstype.uint32)
    t_uint64 = Tensor(np.ones([2, 1, 2, 2]), mstype.uint64)
    with pytest.raises(TypeError):
        net(t_uint16, t_uint8)
    with pytest.raises(TypeError):
        net(t_uint16, t_int8)
    with pytest.raises(TypeError):
        net(t_uint16, t_int16)
    with pytest.raises(TypeError):
        net(t_uint16, t_int32)
    with pytest.raises(TypeError):
        net(t_uint16, t_int64)
    with pytest.raises(TypeError):
        net(t_uint32, t_uint8)
    with pytest.raises(TypeError):
        net(t_uint32, t_int8)
    with pytest.raises(TypeError):
        net(t_uint32, t_int16)
    with pytest.raises(TypeError):
        net(t_uint32, t_int32)
    with pytest.raises(TypeError):
        net(t_uint32, t_int64)
    with pytest.raises(TypeError):
        net(t_uint64, t_uint8)
    with pytest.raises(TypeError):
        net(t_uint64, t_int8)
    with pytest.raises(TypeError):
        net(t_uint64, t_int16)
    with pytest.raises(TypeError):
        net(t_uint64, t_int32)
    with pytest.raises(TypeError):
        net(t_uint64, t_int64)
    with pytest.raises(TypeError):
        net(t_uint16, t_fp16)
    with pytest.raises(TypeError):
        net(t_uint16, t_fp32)
    with pytest.raises(TypeError):
        net(t_uint16, t_fp64)
    with pytest.raises(TypeError):
        net(t_uint32, t_fp16)
    with pytest.raises(TypeError):
        net(t_uint32, t_fp32)
    with pytest.raises(TypeError):
        net(t_uint32, t_fp64)
    with pytest.raises(TypeError):
        net(t_uint64, t_fp16)
    with pytest.raises(TypeError):
        net(t_uint64, t_fp32)
    with pytest.raises(TypeError):
        net(t_uint64, t_fp64)

    with pytest.raises(TypeError):
        tfp(t_uint16)
    with pytest.raises(TypeError):
        tfp(t_uint32)
    with pytest.raises(TypeError):
        tfp(t_uint64)

    with pytest.raises(TypeError):
        tint(t_uint16)
    with pytest.raises(TypeError):
        tint(t_uint32)
    with pytest.raises(TypeError):
        tint(t_uint64)

    bnet = TensorBoolAutoCast()
    with pytest.raises(TypeError):
        bnet(t_uint8)
    with pytest.raises(TypeError):
        bnet(t_int8)
    with pytest.raises(TypeError):
        bnet(t_int16)
    with pytest.raises(TypeError):
        bnet(t_int32)
    with pytest.raises(TypeError):
        bnet(t_int64)
    with pytest.raises(TypeError):
        bnet(t_fp16)
    with pytest.raises(TypeError):
        bnet(t_fp32)
    with pytest.raises(TypeError):
        bnet(t_fp64)
