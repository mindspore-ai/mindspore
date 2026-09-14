#!/usr/bin/env python3
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
Tests for other operations (non-unary/binary), e.g., chunk/gather.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, mint, nn, ops
from mindspore.ops import auto_generate as gen
from tests.mark_utils import arg_mark
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._op_info.op_database import get_op_info, other_op_db, other_op_kbk_db, \
    other_op_error_db, other_op_error_kbk_db

ops_params_dict = {
    'test_other_op_reference_forward' : (['pynative'], other_op_db),
    'test_other_op_reference_backward' : (['pynative'], other_op_db),
    'test_other_op_reference_forward_kbk' : (['kbk'], other_op_kbk_db),
    'test_other_op_reference_backward_kbk' : (['kbk'], other_op_kbk_db),
    'test_other_op_dynamic_forward': (['kbk'], other_op_kbk_db),
    'test_other_op_dynamic_backward': (['kbk'], other_op_kbk_db),
    'test_other_op_error': (['pynative'], other_op_error_db),
    'test_other_op_error_kbk': (['kbk'], other_op_error_kbk_db),
}


class DiagDynamicRankNet(nn.Cell):
    def construct(self, x):
        return mint.diag(x, diagonal=0)


class DiagDynamicRankGradNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.grad = ops.GradOperation()(DiagDynamicRankNet())

    def construct(self, x):
        return self.grad(x)


class MaxUnpool2dDynamicRankIndicesNet(nn.Cell):
    def construct(self, x, indices):
        return mint.nn.functional.max_unpool2d(x, indices, kernel_size=2, stride=2, padding=0)


class CdistNet(nn.Cell):
    def construct(self, x):
        return mint.cdist(x, x, 2.0)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=36.00)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_reference_forward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_reference_forward'][1])
def test_other_op_reference_forward(mode, op_info):
    '''
    Feature: Other operations
    Description: Compare forward.
    Expectation: MindSpore matches Benchmark for outputs.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_reference_backward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_reference_backward'][1])
def test_other_op_reference_backward(mode, op_info):
    '''
    Feature: Other operations
    Description: Compare gradients.
    Expectation: MindSpore matches Benchmark for gradients.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference(grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_reference_forward_kbk'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_reference_forward_kbk'][1])
def test_other_op_reference_forward_kbk(mode, op_info):
    '''
    Feature: Other operations
    Description: Compare forward.
    Expectation: MindSpore matches Benchmark for outputs.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard', mem_peak=36.00)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_reference_backward_kbk'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_reference_backward_kbk'][1])
def test_other_op_reference_backward_kbk(mode, op_info):
    '''
    Feature: Other operations
    Description: Compare gradients.
    Expectation: MindSpore matches Benchmark for gradients.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference(grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_dynamic_forward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_dynamic_forward'][1])
def test_other_op_dynamic_forward(mode, op_info):
    '''
    Feature: Other operations
    Description: Dynamic shape forward.
    Expectation: Outputs match Benchmark.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_dynamic(only_dynamic_shape=True)
    fact.test_op_dynamic(only_dynamic_rank=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ['kbk'])
@pytest.mark.parametrize("op_info", ['mint.nn.functional.interpolate(mode="nearest")-2d'])
def test_other_op_dynamic_interpolate_sequence_cast(mode, op_info):
    '''
    Feature: Other operations.
    Description: Test dynamic sequence scale_factor casting for interpolate.
    Expectation: Run success.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_dynamic(only_dynamic_shape=True)
    fact.test_op_dynamic(only_dynamic_rank=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_dynamic_backward'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_dynamic_backward'][1])
def test_other_op_dynamic_backward(mode, op_info):
    '''
    Feature: Other operations
    Description: Dynamic shape backward.
    Expectation: Gradients match Benchmark.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_dynamic(only_dynamic_shape=True, grad_cmp=True)
    if not fact.op_info.skip_dynamic_rank_backward:
        fact.test_op_dynamic(only_dynamic_rank=True, grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=2048.0)
def test_mint_diag_dynamic_rank_backward_error():
    '''
    Feature: mint.diag.
    Description: Test that backward does not support dynamic rank.
    Expectation: Raise RuntimeError.
    '''
    ms.set_context(mode=ms.GRAPH_MODE)
    grad_net = DiagDynamicRankGradNet()
    grad_net.set_inputs(Tensor(shape=None, dtype=ms.float32))

    with pytest.raises(RuntimeError, match="For 'DiagExt', dynamic rank is not supported in backward"):
        grad_net(Tensor(np.ones((5,), np.float32)))


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard', mem_peak=36.0)
def test_cdist_grad_rank_exceeds_3d_error():
    '''
    Feature: CdistGrad infer.
    Description: Only the internal form created by the backend CdistGrad fission pass (marked with
                 batch_rank) may use the expanded 4D layout, external calls keep the 2D/3D contract.
    Expectation: A rank >= 4 call raises ValueError instead of being passed to the Ascend kernel.
    '''
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend')
    cdist_grad = gen.CdistGrad(p=2.0)
    x3 = Tensor(np.random.rand(4, 8, 16).astype(np.float32))
    assert cdist_grad(x3, x3, x3, x3).shape == (4, 8, 16)
    x4 = Tensor(np.random.rand(2, 4, 8, 16).astype(np.float32))
    with pytest.raises(ValueError, match=r"For primitive\[CdistGrad\], the input_x dim must be in \[2,3\]"):
        cdist_grad(x4, x4, x4, x4)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard', mem_peak=36.0)
def test_cdist_rank_exceeds_3d_error():
    '''
    Feature: Cdist infer.
    Description: The 2D/3D rank check of Cdist has to name the primitive that received the
                 invalid input instead of the gradient operator.
    Expectation: The error message reports Cdist, not the gradient operator.
    '''
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')
    x4 = Tensor(np.random.rand(2, 4, 8, 16).astype(np.float32))
    with pytest.raises(ValueError, match=r"For primitive\[Cdist\], the input_x dim must be in \[2,3\]"):
        CdistNet()(x4)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard', mem_peak=36.0)
def test_mint_max_unpool2d_dynamic_rank_indices():
    '''
    Feature: mint.nn.functional.max_unpool2d.
    Description: The rank of `indices` may be unknown at infer time while the input keeps a dynamic
                 shape, so the spatial dims must not be written with the input rank on the shape
                 vector taken from `indices`.
    Expectation: The output shape is derived from the input through the fallback path, no crash.
    '''
    x_np = np.random.rand(2, 3, 4, 4).astype(np.float32)
    indices_np = np.random.randint(0, 16, size=(2, 3, 4, 4)).astype(np.int32)
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend')
    expect = mint.nn.functional.max_unpool2d(Tensor(x_np), Tensor(indices_np), kernel_size=2, stride=2,
                                             padding=0)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')
    net = MaxUnpool2dDynamicRankIndicesNet()
    net.set_inputs(Tensor(shape=(None, None, None, None), dtype=ms.float32),
                   Tensor(shape=None, dtype=ms.int32))
    out = net(Tensor(x_np), Tensor(indices_np))
    assert out.shape == expect.shape
    np.testing.assert_allclose(out.asnumpy(), expect.asnumpy(), rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard', mem_peak=1024.0)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_error'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_error'][1])
def test_other_op_error(mode, op_info):
    '''
    Feature: other operations
    Description: Test other op error cases.
    Expectation: Run success without error.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_other_op_error()

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard', mem_peak=1024.0)
@pytest.mark.parametrize("mode", ops_params_dict['test_other_op_error_kbk'][0])
@pytest.mark.parametrize("op_info", ops_params_dict['test_other_op_error_kbk'][1])
def test_other_op_error_kbk(mode, op_info):
    '''
    Feature: Binary operations
    Description: Test binary op error cases.
    Expectation: Run success without error.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_other_op_error()
