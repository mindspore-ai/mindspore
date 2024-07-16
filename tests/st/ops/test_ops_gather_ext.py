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


"""test gather"""

import numpy as np
import pytest

from mindspore import ops, Tensor, jit, JitConfig, context
from mindspore.common.api import _pynative_executor
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
import time


def GenInputData(np_data_type, shape=(3, 4, 5)):
    """GenInputData"""
    size = 1
    for s in shape:
        size *= s
    data = np.arange(size).reshape(*shape).astype(np_data_type)
    return Tensor(data)


def GenIndexData(np_data_type):
    """GenIndexData"""
    indices = np.array([[[3, 3, 2, 2],
                         [2, 2, 0, 2],
                         [0, 3, 0, 2]],

                        [[0, 1, 2, 3],
                         [2, 2, 2, 3],
                         [2, 0, 2, 1]]]).astype(np_data_type)
    return Tensor(indices)

def GenDim():
    """GenDim"""
    return 1

def GenExpectResult(np_data_type):
    """GenExpectResult"""
    expect = np.array([[[15, 16, 12, 13],
                        [10, 11, 2, 13],
                        [0, 16, 2, 13]],

                       [[20, 26, 32, 38],
                        [30, 31, 32, 38],
                        [30, 21, 32, 28]]]).astype(np_data_type)
    return expect


def GenGradOut(np_data_type):
    """GenGradOut"""
    expect = np.array([[[1., 0., 2., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [1., 1., 1., 3., 0.],
                        [1., 2., 0., 0., 0.]],

                       [[1., 1., 0., 0., 0.],
                        [0., 1., 0., 1., 0.],
                        [2., 1., 3., 0., 0.],
                        [0., 0., 0., 2., 0.]],

                       [[0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]]]).astype(np_data_type)
    return expect


def call_gather(x, dim, indices):
    """call_gather"""
    out = ops.function.array_func.gather_ext(x, dim, indices)
    return out


def gather_ext_backward_func(x, dim, indices):
    """gather_ext_backward_func"""
    return ops.grad(call_gather)(x, dim, indices)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
@pytest.mark.parametrize('input_dtype', [np.float32])
@pytest.mark.parametrize('index_dtype', [np.int64])
def test_gather_ext_static_shape(mode, input_dtype, index_dtype):
    """
    Feature: Test gather with static shape in graph and pynative mode.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    ms_data = GenInputData(input_dtype)
    ms_indices = GenIndexData(index_dtype)
    dim = GenDim()

    if mode == 'pynative':
        ms_out = call_gather(ms_data, dim, ms_indices)
    elif mode == 'KBK':
        ms_out = (jit(call_gather, jit_config=JitConfig(jit_level="O0")))(ms_data, dim, ms_indices)
    else:
        ms_out = (jit(call_gather, jit_config=JitConfig(jit_level="O2")))(ms_data, dim, ms_indices)

    expect = GenExpectResult(input_dtype)
    assert np.allclose(ms_out.asnumpy(), expect, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
def test_gather_ext_dynamic_shape():
    """
    Feature: Test gather with dynamic shape in graph mode.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    ms_data1 = GenInputData(np.float32, (3, 4, 5))
    ms_indices1 = Tensor(np.random.randint(3, size=(14, 2, 2)))
    dim1 = 0

    ms_data2 = GenInputData(np.float32, (3, 7, 8, 3))
    ms_indices2 = Tensor(np.random.randint(8, size=(2, 6, 4, 3)))
    dim2 = 2
    TEST_OP(call_gather, [[ms_data1, dim1, ms_indices1], [ms_data2, dim2, ms_indices2]], 'gather_d')


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_gather_ext_vmap(param_jit_level):
    """
    Feature: Test gather with vmap.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    def _foreach_run(x, dim, index, batch_axis):
        out = []
        for i in range(x.shape[batch_axis]):
            if batch_axis == -1:
                input_inner = x[..., i]
                index_inner = index[..., i]
            else:
                input_inner = x[i, ...]
                index_inner = index[i, ...]
            out.append(call_gather(input_inner, dim, index_inner))
        out = ops.Stack()(out)
        return out

    context.set_context(jit_level=param_jit_level)
    ms_data = GenInputData(np.float32, (4, 5, 6))
    ms_indices = Tensor(np.random.randint(4, size=(4, 5, 6)))
    dim = GenDim()
    batch_axis = -1
    gather_ext_vmap_func = ops.vmap(call_gather, in_axes=(batch_axis, None, batch_axis), out_axes=0)
    ms_out = gather_ext_vmap_func(ms_data, dim, ms_indices)
    expect = _foreach_run(ms_data, dim, ms_indices, batch_axis)
    assert np.allclose(ms_out.asnumpy(), expect.asnumpy(), rtol=1e-4)

    batch_axis = 0
    gather_ext_vmap_func = ops.vmap(call_gather, in_axes=(batch_axis, None, batch_axis), out_axes=0)
    ms_out = gather_ext_vmap_func(ms_data, dim, ms_indices)
    expect = _foreach_run(ms_data, dim, ms_indices, batch_axis)
    assert np.allclose(ms_out.asnumpy(), expect.asnumpy(), rtol=1e-4)



# @pytest.mark.parametrize('batch', [8, 16, 32, 64, 128])
def _test_gather_ext_vmap_perf(batch):
    """
    Feature: Test gather with vmap perf.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    vmap_func = ops.vmap(ops.vmap(call_gather, in_axes=(0, None, 0), out_axes=0), in_axes=(0, None, 0), out_axes=0)

    # @jit
    def _foreach_run(inputs, dim, indices):
        out = []
        for inputs_inner, indices_inner in zip(inputs, indices):
            out_inner = []
            for x, index in zip(inputs_inner, indices_inner):
                out_inner.append(call_gather(x, dim, index))
            out.append(out_inner)
        return out

    ops.Abs()(Tensor(5.0))
    _pynative_executor.sync()
    ms_data = GenInputData(np.float32, (batch, 4, 4, 4))
    ms_indices = Tensor(np.random.randint(4, size=(batch, 4, 4, 4)))
    dim = 1
    ms_data_ori = []
    ms_indices_ori = []
    for i in range(ms_data.shape[0]):
        ms_data_ori_inner = []
        ms_indices_ori_inner = []
        for j in range(ms_data.shape[1]):
            ms_data_ori_inner.append(ms_data[i, j, ...])
            ms_indices_ori_inner.append(ms_indices[i, j, ...])
        ms_data_ori.append(ms_data_ori_inner)
        ms_indices_ori.append(ms_indices_ori_inner)

    _pynative_executor.sync()
    run_times = 100
    for i in range(15):
        ms_out = vmap_func(ms_data, dim, ms_indices)
        ori_out_list = _foreach_run(ms_data_ori, dim, ms_indices_ori)
    _pynative_executor.sync()

    start = time.time()
    for _ in range(run_times):
        ms_out = vmap_func(ms_data, dim, ms_indices)
    _pynative_executor.sync()
    end = time.time()
    vmap_duration = end - start

    start = time.time()
    for _ in range(run_times):
        ori_out_list = _foreach_run(ms_data_ori, dim, ms_indices_ori)
    _pynative_executor.sync()
    end = time.time()
    foreach_duration = end - start

    ori_out_inner = []
    for i in range(ms_data.shape[0]):
        ori_out_inner.append(ops.Stack()(ori_out_list[i]))
    ori_out = ops.Stack()(ori_out_inner)
    assert np.allclose(ms_out.asnumpy(), ori_out.asnumpy(), rtol=1e-4)

    print(f"Testing vmap perf with batch={batch}:")
    print(f"foreach_duration: {foreach_duration / run_times}")
    print(f"vmap_duration: {vmap_duration / run_times}")
    print(f"improve_times: {foreach_duration / vmap_duration}\n")


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", ['pynative', 'GE', 'KBK'])
def test_gather_ext_grad(mode):
    """
    Feature: Test gather with backward.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    ms_data = GenInputData(np.float32)
    ms_indices = GenIndexData(np.int64)
    expect = GenGradOut(np.float32)
    dim = GenDim()

    if mode == 'pynative':
        ms_out = gather_ext_backward_func(ms_data, dim, ms_indices)
    elif mode == 'KBK':
        ms_out = (jit(gather_ext_backward_func, jit_config=JitConfig(jit_level="O0")))(ms_data, dim, ms_indices)
    else:
        ms_out = (jit(gather_ext_backward_func, jit_config=JitConfig(jit_level="O2")))(ms_data, dim, ms_indices)
    assert np.allclose(ms_out.asnumpy(), expect, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_gather_ext_unmatch_shape():
    """
    Feature: Test gather .
    Description: index shape is not equal to input shape when axis != dim.
    Expectation: return the correct value.
    """
    shape = (3, 2, 3, 2, 3)
    ms_data = Tensor(np.arange(3 * 2 * 3 * 2 * 3).reshape(shape).astype(np.float32))
    ms_indices = Tensor([[[[[1, 1]],
                           [[1, 0]]]],
                         [[[[0, 1]],
                           [[1, 1]]]]])
    expect_fw = [[[[[6., 7.]],
                   [[6., 1.]]]],
                 [[[[36., 43.]],
                   [[42., 43.]]]]]
    dim = 2

    fw_out = call_gather(ms_data, dim, ms_indices)
    assert np.allclose(fw_out.asnumpy(), expect_fw, rtol=1e-4)

    expect_bw = [[[[[0., 1., 0.], [0., 0., 0.]],
                   [[2., 1., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]]],
                  [[[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]]]],
                 [[[[1., 0., 0.], [0., 0., 0.]],
                   [[1., 2., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]]],
                  [[[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]]]],
                 [[[[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]]],
                  [[[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., 0., 0.]]]]]
    bw_out = gather_ext_backward_func(ms_data, dim, ms_indices)
    assert np.allclose(bw_out.asnumpy(), expect_bw, rtol=1e-4)
