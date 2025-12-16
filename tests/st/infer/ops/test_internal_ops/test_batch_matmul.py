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
Tests for batch matmul operator.
"""
import os
import numpy as np
import pytest

import mindspore as ms
from mindspore import context
from mindspore import Profiler
from mindspore.common.np_dtype import bfloat16
from tests.mark_utils import arg_mark


class BatchMatMulCustom(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.BatchMatMul(ta, tb)

    def construct(self, i0, i1):
        return self.net(i0, i1)


def compare(out, expect, dtype):
    if dtype == ms.float16:
        limit = 0.004
    elif dtype == ms.bfloat16:
        limit = 0.03
    else:
        limit = 0.004  # default limit

    out_flatten = out.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum((np.abs(out_flatten - expect_flatten) /
                     np.abs(expect_flatten) > limit).astype(np.int32))
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        #print("[FAILED] err_cnt = ", err_cnt, "/", limit_cnt)
        return False
    #print("[SUCCESS] err_cnt = ", err_cnt, "/", limit_cnt)
    return True


def _test_batch_matmul(m, k, n, b0=0, b1=0, trans_a=False, trans_b=False, mstype=ms.float16, profiling=False):
    if b0 == 0 or b1 == 0:
        raise ValueError("this is batch matmul testcase, b can't be 0")

    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.float32 == mstype:
        np_type = np.float32
    elif ms.bfloat16 == mstype:
        np_type = bfloat16
    else:
        np_type = np.float16  # default

    if trans_a:
        i0_host = np.random.normal(0.0, 0.5, size=[b0, k, m]).astype(np_type)
    else:
        i0_host = np.random.normal(0.0, 0.5, size=[b0, m, k]).astype(np_type)

    if trans_b:
        i1_host = np.random.normal(0.0, 0.5, size=[b1, n, k]).astype(np_type)
    else:
        i1_host = np.random.normal(0.0, 0.5, size=[b1, k, n]).astype(np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    if not trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32)
    elif not trans_a and trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32.transpose(0, 2, 1))
    elif trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32.transpose(0, 2, 1), i1_host_fp32)
    else:  # trans_a and trans_b
        expect = np.matmul(i0_host_fp32.transpose(0, 2, 1),
                           i1_host_fp32.transpose(0, 2, 1))

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)

    net = BatchMatMulCustom(trans_a, trans_b)

    if profiling:
        profiler = Profiler(start_profile=False, output_path="profiler")
        profiler.start()
        for _ in range(50):
            output = net(input1, input2)
        profiler.stop()
        profiler.analyse()
        return

    output = net(input1, input2)
    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = compare(expect, output_np, mstype)
    assert res, "matmul compare fail."


def get_tensor_with_different_dim(b0, b1, param1, param2, dim, transpose, np_type):
    if dim == 2:
        if transpose:
            return np.random.normal(0.0, 0.5, size=[param2, param1]).astype(np_type)
        return np.random.normal(0.0, 0.5, size=[param1, param2]).astype(np_type)
    if dim == 3:
        if transpose:
            return np.random.normal(0.0, 0.5, size=[b0, param2, param1]).astype(np_type)
        return np.random.normal(0.0, 0.5, size=[b0, param1, param2]).astype(np_type)
    if dim == 4:
        if transpose:
            return np.random.normal(0.0, 0.5, size=[b0, b1, param2, param1]).astype(np_type)
        return np.random.normal(0.0, 0.5, size=[b0, b1, param1, param2]).astype(np_type)
    raise ValueError("wrong dim value")


def batch_matmul(m, k, n, dim_type, b0=0, b1=0, trans_a=False, trans_b=False, mstype=ms.float16, profiling=False):
    if b0 == 0 or b1 == 0:
        raise ValueError("this is batch matmul testcase, b can't be 0")
    if (dim_type is None or dim_type["x_dim"] > 4
            or dim_type["x_dim"] < 2 or dim_type["weight_dim"] < 2 or dim_type["weight_dim"] > 3):
        raise ValueError("dim_type parameter is out of range")
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.float32 == mstype:
        np_type = np.float32
    elif ms.bfloat16 == mstype:
        np_type = bfloat16
    else:
        np_type = np.float16  # default

    i0_host = get_tensor_with_different_dim(
        b0, b1, m, k, dim_type["x_dim"], trans_a, np_type)
    i1_host = get_tensor_with_different_dim(
        b1, 0, k, n, dim_type["weight_dim"], trans_b, np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)

    if not trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32)
    elif not trans_a and trans_b:
        if dim_type["weight_dim"] == 3:
            expect = np.matmul(i0_host_fp32, i1_host_fp32.transpose(0, 2, 1))
        else:
            expect = np.matmul(i0_host_fp32, i1_host_fp32.T)
    elif trans_a and not trans_b:
        if dim_type["x_dim"] == 4:
            expect = np.matmul(i0_host_fp32.transpose(
                0, 1, 3, 2), i1_host_fp32)
        elif dim_type["x_dim"] == 3:
            expect = np.matmul(i0_host_fp32.transpose(0, 2, 1), i1_host_fp32)
        else:
            expect = np.matmul(i0_host_fp32.T, i1_host_fp32)
    else:  # trans_a and trans_b
        if dim_type["x_dim"] == 4:
            i0_host_fp32_t = i0_host_fp32.transpose(0, 1, 3, 2)
        elif dim_type["x_dim"] == 3:
            i0_host_fp32_t = i0_host_fp32.transpose(0, 2, 1)
        else:
            i0_host_fp32_t = i0_host_fp32.T

        if dim_type["weight_dim"] == 3:
            i1_host_fp32_t = i1_host_fp32.transpose(0, 2, 1)
        else:
            i1_host_fp32_t = i1_host_fp32.T

        expect = np.matmul(i0_host_fp32_t, i1_host_fp32_t)

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)

    net = BatchMatMulCustom(trans_a, trans_b)

    if profiling:
        for _ in range(50):
            output = net(input1, input2)
        return

    output = net(input1, input2)
    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = compare(expect, output_np, mstype)
    assert res, "batch matmul compare fail."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('ms_dtype', [ms.bfloat16])
@pytest.mark.parametrize('shape', [[16, 32, 64]])
@pytest.mark.parametrize('batch_size_list', [[1, 1], [5, 1], [8, 8]])
@pytest.mark.parametrize('trans_b', [True, False])
def test_batch_matmul_small(ms_dtype, shape, batch_size_list, trans_b):
    """
    Feature: test BatchMatmul with small shape
    Description: test BatchMatmul. One of the input's batch dim must be equal to another input's peer batch dim, or
    be equal to 1, or be empty.
    Expectation: the result is correct
    """
    m, k, n = shape
    b0, b1 = batch_size_list
    _test_batch_matmul(m, k, n, b0, b1, trans_a=False,
                       trans_b=trans_b, mstype=ms_dtype, profiling=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_dtype', [ms.bfloat16])
@pytest.mark.parametrize('shape', [[256, 256, 256]])
@pytest.mark.parametrize('batch_size_list', [[1, 1], [5, 1], [8, 8]])
@pytest.mark.parametrize('trans_b', [True, False])
def test_batch_matmul_medium(ms_dtype, shape, batch_size_list, trans_b):
    """
    Feature: test BatchMatmul with medium shape
    Description: test BatchMatmul. One of the input's batch dim must be equal to another input's peer batch dim, or
    be equal to 1, or be empty.
    Expectation: the result is correct
    """
    m, k, n = shape
    b0, b1 = batch_size_list
    _test_batch_matmul(m, k, n, b0, b1, trans_a=False,
                       trans_b=trans_b, mstype=ms_dtype, profiling=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_dtype', [ms.bfloat16])
@pytest.mark.parametrize('shape', [[1024, 4096, 4096]])
@pytest.mark.parametrize('batch_size_list', [[1, 1], [5, 1], [8, 8]])
@pytest.mark.parametrize('trans_b', [True, False])
def test_batch_matmul_large(ms_dtype, shape, batch_size_list, trans_b):
    """
    Feature: test BatchMatmul with large shape
    Description: test BatchMatmul. One of the input's batch dim must be equal to another input's peer batch dim, or
    be equal to 1, or be empty.
    Expectation: the result is correct
    """
    m, k, n = shape
    b0, b1 = batch_size_list
    _test_batch_matmul(m, k, n, b0, b1, trans_a=False,
                       trans_b=trans_b, mstype=ms_dtype, profiling=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_dtype', [ms.float16])
@pytest.mark.parametrize('shape', [[16, 32, 64], [256, 256, 256]])
@pytest.mark.parametrize('batch_size_list', [[1, 1], [5, 1], [8, 8]])
@pytest.mark.parametrize('trans_b', [True, False])
@pytest.mark.parametrize('dim_type', [{"x_dim": 3, "weight_dim": 3}, {"x_dim": 3, "weight_dim": 2}])
def test_batch_matmul_3dx_ndw_fp16(ms_dtype, shape, batch_size_list, trans_b, dim_type):
    """
    Feature: test batchMatmul
    Description: test batchMatmul. One of the input's batch dim must be equal to another input's peer batch dim, or
    be equal to 1, or be empty.
    Expectation: the result is correct
    """
    m, k, n = shape
    b0, b1 = batch_size_list
    batch_matmul(m, k, n, dim_type, b0, b1, trans_a=False,
                 trans_b=trans_b, mstype=ms_dtype, profiling=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_dtype', [ms.bfloat16])
@pytest.mark.parametrize('shape', [[1024, 4096, 4096]])
@pytest.mark.parametrize('batch_size_list', [[1, 1], [5, 1], [8, 8]])
@pytest.mark.parametrize('trans_b', [True, False])
@pytest.mark.parametrize('dim_type', [{"x_dim": 3, "weight_dim": 3}, {"x_dim": 3, "weight_dim": 2}])
def test_batch_matmul_3dx_ndw_bf16(ms_dtype, shape, batch_size_list, trans_b, dim_type):
    """
    Feature: test batchMatmul
    Description: test batchMatmul. One of the input's batch dim must be equal to another input's peer batch dim, or
    be equal to 1, or be empty.
    Expectation: the result is correct
    """
    m, k, n = shape
    b0, b1 = batch_size_list
    batch_matmul(m, k, n, dim_type, b0, b1, trans_a=False,
                 trans_b=trans_b, mstype=ms_dtype, profiling=False)


def batch_matmul_swft(m, k, n, dim_type, b0=0, b1=0, trans_a=False, trans_b=False, mstype=ms.float16, profiling=False):
    if b0 == 0 or b1 == 0:
        raise ValueError("this is batch matmul testcase, b can't be 0")
    if (dim_type is None or dim_type["x_dim"] > 4 or dim_type["x_dim"] < 2
            or dim_type["weight_dim"] < 2 or dim_type["weight_dim"] > 3):
        raise ValueError("dim_type parameter is out of range")
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.float32 == mstype:
        np_type = np.float32
    elif ms.bfloat16 == mstype:
        np_type = bfloat16
    else:
        np_type = np.float16  # default

    i0_host = get_tensor_with_different_dim(
        b0, b1, m, k, dim_type["x_dim"], trans_a, np_type)
    i1_host = get_tensor_with_different_dim(
        b1, 0, k, n, dim_type["weight_dim"], trans_b, np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)

    if not trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32)
    elif not trans_a and trans_b:
        if dim_type["weight_dim"] == 3:
            expect = np.matmul(i0_host_fp32, i1_host_fp32.transpose(0, 2, 1))
        else:
            expect = np.matmul(i0_host_fp32, i1_host_fp32.T)
    elif trans_a and not trans_b:
        if dim_type["x_dim"] == 4:
            expect = np.matmul(i0_host_fp32.transpose(
                0, 1, 3, 2), i1_host_fp32)
        elif dim_type["x_dim"] == 3:
            expect = np.matmul(i0_host_fp32.transpose(0, 2, 1), i1_host_fp32)
        else:
            expect = np.matmul(i0_host_fp32.T, i1_host_fp32)
    else:  # trans_a and trans_b
        if dim_type["x_dim"] == 4:
            i0_host_fp32_t = i0_host_fp32.transpose(0, 1, 3, 2)
        elif dim_type["x_dim"] == 3:
            i0_host_fp32_t = i0_host_fp32.transpose(0, 2, 1)
        else:
            i0_host_fp32_t = i0_host_fp32.T

        if dim_type["weight_dim"] == 3:
            i1_host_fp32_t = i1_host_fp32.transpose(0, 2, 1)
        else:
            i1_host_fp32_t = i1_host_fp32.T

        expect = np.matmul(i0_host_fp32_t, i1_host_fp32_t)

    input1 = ms.Tensor(np.expand_dims(i0_host_fp32, 0), mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)

    net = BatchMatMulCustom(trans_a, trans_b)

    if profiling:
        for _ in range(50):
            output = net(input1, input2)
        return

    output = net(input1, input2)
    output_fp32 = output.astype(ms.float32).squeeze(0)
    output_np = output_fp32.asnumpy()
    res = compare(expect, output_np, mstype)
    assert res, "batch matmul compare fail."


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('batch_size', [16, 32])
@pytest.mark.parametrize('trans_b', [True, False])
def test_batch_matmul_swft(batch_size, trans_b):
    """
    Feature: test batch_matmul with swft kernels.
    Description: test batch_matmul.
    Expectation: the result is correct
    """
    m, k, n = [1, 512, 128]
    k, n = (k, n) if trans_b else (n, k)
    batch_matmul_swft(m, k, n, {"x_dim": 3, "weight_dim": 3}, batch_size, batch_size,
                      trans_a=False, trans_b=trans_b, mstype=ms.float16, profiling=False)
