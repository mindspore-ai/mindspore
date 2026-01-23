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
""" tests_custom_pyboost_ascend """

import pytest
import numpy as np
import mindspore as ms
from mindspore._c_expression import typing
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_pyboost_atb_swiglu():
    """
    Feature: CustomOpBuilder.
    Description: Custom atb op.
    Expectation: success.
    """
    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("atb_swiglu", "jit_test_files/atb_swiglu.cpp", enable_atb=True).load()
    # the second dim of x should be >= 32
    x = np.array([[0.561, 0.684, 0.329, 0.8447, 0.2815, 0.0716, 0.3472, 0.04404,
                   0.9565, 0.9033, 0.3567, 0.33, 0.2467, 0.2993, 0.0109, 0.9243,
                   0.2163, 0.4355, 0.4707, 0.9463, 0.5156, 0.978, 0.815, 0.247,
                   0.7153, 0.677, 0.9263, 0.665, 0.353, 0.0239, 0.4363, 0.9097],
                  [0.9585, 0.1242, 0.05566, 0.642, 0.5103, 0.658, 0.704, 0.4739,
                   0.299, 0.1958, 0.2349, 0.10657, 0.2134, 0.1458, 0.4458, 0.2399,
                   0.6626, 0.4255, 0.5674, 0.5454, 0.3523, 0.5435, 0.03458, 0.912,
                   0.3064, 0.9287, 0.8633, 0.2822, 0.652, 0.1549, 0.6426, 0.004536]], dtype=np.float16)
    expect = np.array([[0.0773, 0.198, 0.0901, 0.559, 0.0827, 0.03625, 0.1658, 0.005558,
                        0.4944, 0.435, 0.1943, 0.1277, 0.0489, 0.00411, 0.002392, 0.602],
                       [0.459, 0.02806, 0.01624, 0.2295, 0.1123, 0.2357, 0.0163, 0.2664,
                        0.0526, 0.0998, 0.1132, 0.01584, 0.07697, 0.01211, 0.1747, 0.000609]], dtype=np.float16)
    output = my_ops.npu_swiglu(ms.Tensor(x), -1)
    assert np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=20.00)
def test_pyboost_aclnn_quant_batch_matmul_nz_format():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE)
    my_ops = CustomOpBuilder("quant_batch_matmul", ["jit_test_files/pyboost_aclnn_quant_batch_matmul.cpp"],
                             backend="Ascend").load()

    batch = 2
    m = 128
    k = 256
    n = 128
    x1 = np.random.randint(-5, 5, size=(batch, m, k)).astype(np.int8)
    x2 = np.random.randint(-5, 5, size=(batch, k, n)).astype(np.int8)
    scale = np.ones([n]).astype(np.float32)
    expected = np.matmul(x1.astype(np.int32), x2.astype(np.int32)) * scale

    ms_x1 = ms.Tensor(x1)
    ms_x2 = ms.Tensor(x2)
    # 29 -> FRACTAL_NZ
    ms_x2 = ms.ops.auto_generate.format_cast(ms_x2, 29)
    ms_scale = ms.Tensor(scale)
    # 45 -> output_dtype: ms.bfloat16
    output = my_ops.quant_batch_matmul(ms_x1, ms_x2, ms_scale, None, None, None, False, False, "FRACTAL_NZ", 45)
    assert np.allclose(expected, output.astype(ms.float32).asnumpy(), 0.01)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=18.0)
def test_pyboost_atb_rope():
    """
    Feature: CustomOpBuilder.
    Description: Custom atb op.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ntokens = 4
    head_size = 8
    hiddenSizeQ = 16
    hiddenSizeK = 16

    cosCacheNeox = None
    sinCacheNeox = None
    cosCache = None
    sinCache = None
    sequenceLength = None
    previousTokenCount = -1

    def run_bencmkark(my_ops, positions, query, key, head_size, cos_sin_cache, is_neox_style):
        nonlocal cosCacheNeox
        nonlocal sinCacheNeox
        nonlocal cosCache
        nonlocal sinCache
        nonlocal sequenceLength
        nonlocal previousTokenCount
        if cosCache is None or sinCache is None:
            cosSinChunks = cos_sin_cache.chunk(2, -1)
            cosCache = cosSinChunks[0].repeat_interleave(2, 1)
            sinCache = cosSinChunks[1].repeat_interleave(2, 1)
            cosCacheNeox = cosSinChunks[0].repeat((1, 2))
            sinCacheNeox = cosSinChunks[1].repeat((1, 2))
        flatPositions = positions.flatten()
        currentTokenCount = flatPositions.shape[0]
        cos = cosCacheNeox.index_select(0, flatPositions) if is_neox_style else cosCache.index_select(0, flatPositions)
        sin = sinCacheNeox.index_select(0, flatPositions) if is_neox_style else sinCache.index_select(0, flatPositions)
        if sequenceLength is None or currentTokenCount != previousTokenCount:
            previousTokenCount = currentTokenCount
            sequenceLength = ms.Tensor([1], dtype=ms.int32)
        rotaryCoeff = 2 if is_neox_style else head_size
        my_ops.rope_native_atb(query, key, cos, sin, sequenceLength, rotaryCoeff)

    my_ops = CustomOpBuilder("atb_rope", "jit_test_files/atb_rope.cpp", enable_atb=True).load()
    np.random.seed(100)
    const_positions = ms.Tensor(np.array([0, 2, 4, 6], dtype=np.int32))
    np_query = np.random.rand(ntokens, hiddenSizeQ).astype(np.float16)
    np_key = np.random.rand(ntokens, hiddenSizeK).astype(np.float16)
    const_cos_sin_cache = ms.Tensor(np.random.rand(ntokens * 2, head_size).astype(np.float32))

    run_query = ms.Tensor(np_query)
    run_key = ms.Tensor(np_key)
    benchmark_query = ms.Tensor(np_query)
    benchmark_key = ms.Tensor(np_key)

    my_ops.npu_rope(const_positions, run_query, run_key, head_size, const_cos_sin_cache, False)
    run_bencmkark(my_ops, const_positions, benchmark_query, benchmark_key, head_size, const_cos_sin_cache, False)
    my_ops.npu_rope(const_positions, run_query, run_key, head_size, const_cos_sin_cache, True)
    run_bencmkark(my_ops, const_positions, benchmark_query, benchmark_key, head_size, const_cos_sin_cache, True)
    assert np.allclose(run_query.asnumpy(), benchmark_query.asnumpy(), 1e-3, 1e-3)
    assert np.allclose(run_key.asnumpy(), benchmark_key.asnumpy(), 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_pyboost_asdsip_fft():
    """
    Feature: CustomOpBuilder.
    Description: Custom asdsip op.
    Expectation: success.
    """
    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("asdsip_fftc2c", "jit_test_files/asdsip_fftc2c.cpp", enable_asdsip=True).load()
    # 1d
    input_np = np.random.rand(2, 16)
    real_np = input_np.astype(np.float32)
    imag_np = input_np.astype(np.float32)
    complex_np = real_np + 1j * imag_np
    input_tensor = ms.Tensor(complex_np, dtype=ms.dtype.complex64)
    output_tensor = my_ops.fft_1d(input_tensor, 16, 2)
    output_np = np.fft.fft(complex_np)
    assert np.allclose(output_tensor.asnumpy(), output_np, 1e-3, 1e-3)

    # 2d
    input_np = np.random.rand(2, 16, 2)
    real_np = input_np.astype(np.float32)
    imag_np = input_np.astype(np.float32)
    complex_np = real_np + 1j * imag_np
    input_tensor = ms.Tensor(complex_np, dtype=ms.dtype.complex64)
    output_tensor = my_ops.fft_2d(input_tensor, 16, 2, 2)
    output_np = np.fft.fft2(complex_np, s=complex_np.shape[-2:], axes=range(-2, 0))
    assert np.allclose(output_tensor.asnumpy(), output_np, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_pyboost_aclnn():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("aclnn_op", ['jit_test_files/pyboost_aclnn_sum.cpp'],
                             backend="Ascend").load()
    x = np.random.rand(4, 5, 6).astype(np.float32)
    expect = np.sum(np.abs(x), 1, keepdims=True)
    output = my_ops.npu_abs_reduce_sum(ms.Tensor(x), (1,), True, None)
    assert np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_pyboost_aclnn_arg_min():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("aclnn_op_2", ['jit_test_files/pyboost_aclnn_argmin.cpp'],
                             backend="Ascend").load()

    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    expect = np.argmin(x, 0)
    output = my_ops.npu_arg_min(ms.Tensor(x), 0, False)
    assert np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_pyboost_aclnn_batch_norm():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("aclnn_op_3", ['jit_test_files/pyboost_aclnn_batch_norm.cpp'],
                             backend="Ascend").load()

    x = ms.Tensor((3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32))
    scale = ms.Tensor(np.ones(2).astype(np.float32))
    bias = ms.Tensor(np.ones(2).astype(np.float32))
    mean = ms.Tensor(np.ones(2).astype(np.float32))
    variance = ms.Tensor(np.ones(2).astype(np.float32))

    expect = np.array([2.99999]).repeat(16, axis=0).astype(np.float32).reshape((2, 2, 1, 4))

    output = my_ops.npu_batch_norm(x, scale, bias, mean, variance, False, 0.1, 1e-5)[0]
    assert np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=10.0)
def test_pyboost_aclnn_cast():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("aclnn_op_4", ['jit_test_files/pyboost_aclnn_cast.cpp'],
                             backend="Ascend").load()

    x = np.random.randn(1280, 1280).astype(np.float16)
    dst_type_id = typing.type_to_type_id(ms.dtype.float32)
    output = my_ops.npu_cast(ms.Tensor(x), dst_type_id)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (1280, 1280)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_pyboost_aclnn_avg_pool_2d():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("aclnn_op_5", ['jit_test_files/pyboost_aclnn_avg_pool_2d.cpp'],
                             backend="Ascend").load()

    image = ms.Tensor(np.array([[[4.1702e-1, 7.2032e-1, 1.1437e-4, 3.0223e-1],
                                 [1.4676e-1, 9.2339e-2, 1.8626e-1, 3.4556e-1],
                                 [3.9677e-1, 5.3882e-1, 4.1919e-1, 6.8522e-1],
                                 [2.0445e-1, 8.7812e-1, 2.7338e-2, 6.7047e-1]]]).astype(np.float32))

    output = my_ops.npu_avgpool2d(image, (2, 2), (2, 2), (1, 1), False, True, 0, False)
    expected = np.array([[[0.1043, 0.1801, 0.0756],
                          [0.1359, 0.3092, 0.2577],
                          [0.0511, 0.2264, 0.1676]]]).astype(np.float32)

    assert np.allclose(output.asnumpy(), expected, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_pyboost_tensor_api():
    """
    Feature: CustomOpBuilder.
    Description: Custom op use tensor api.
    Expectation: success.
    """
    ms.set_device("Ascend")
    my_ops = CustomOpBuilder("tensor_api", "jit_test_files/tensor_api.cpp", backend="Ascend").load()

    x = ms.Tensor(np.random.random((3, 4, 5)).astype(np.float16))
    x_slice = x[:, 1:3, :]
    out = my_ops.reshape_fp32(x_slice, [-1, 5])
    expect = x_slice.reshape((-1, 5)).astype(ms.float32)
    assert np.allclose(out.asnumpy(), expect.asnumpy(), 1e-3, 1e-3)

    assert np.allclose(my_ops.tensor_int(100, "int32").asnumpy(), np.array([100], np.int32), 1e-3, 1e-3)
    assert np.allclose(my_ops.tensor_double(3.14, "float16").asnumpy(), np.array([3.14], np.float16), 1e-3, 1e-3)
    assert np.allclose(my_ops.tensor_int_list([1, 2, 3], "int64").asnumpy(), np.array([1, 2, 3], np.int64), 1e-3, 1e-3)
    assert np.allclose(my_ops.tensor_double_list(
        [1.1, 2.2, 3.3], "float32").asnumpy(), np.array([1.1, 2.2, 3.3], np.float32), 1e-3, 1e-3)
    assert np.allclose(my_ops.ones([3, 4], "float16").asnumpy(), np.ones([3, 4], np.float16), 1e-3, 1e-3)
    assert np.allclose(my_ops.zeros([3, 4], "float32").asnumpy(), np.zeros([3, 4], np.float32), 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_pyboost_asdsip_fft_wrong():
    """
    Feature: CustomOpBuilder.
    Description: Custom asdsip op.
    Expectation: success.
    """
    ms.set_device("Ascend")
    # raise RuntimeError when enable_asdsip is False.
    with pytest.raises(RuntimeError):
        CustomOpBuilder("asdsip_fftc2c", "jit_test_files/asdsip_fftc2c.cpp").load()
    # raise RuntimeError when function name is wrong.
    with pytest.raises(RuntimeError):
        my_ops = CustomOpBuilder("asdsip_fftc2c", "jit_test_files/asdsip_fftc2c.cpp", enable_asdsip=True).load()
        input_np = np.random.rand(2, 16)
        real_np = input_np.astype(np.float32)
        imag_np = input_np.astype(np.float32)
        complex_np = real_np + 1j * imag_np

        input_tensor = ms.Tensor(complex_np, dtype=ms.dtype.complex64)
        output_tensor = my_ops.wrong_fft_1d(input_tensor, 16, 2)
        output_np = np.fft.fft(complex_np)
        assert np.allclose(output_tensor.asnumpy(), output_np, 1e-3, 1e-3)
