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
"""test hccl all_gather_into_tensor and reduce_scatter_tensor grad with 2p"""
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.ops.communication import init_process_group, set_comm_ops_inplace
from mindspore.ops.communication import all_gather_into_tensor, reduce_scatter_tensor
from mindspore.ops.communication import get_rank, get_world_size
from mindspore import value_and_grad
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init_process_group()
set_comm_ops_inplace(False)
this_rank = get_rank()
size = get_world_size()


if size != 2:
    raise RuntimeError("Group size should be 2 exactly.")


class AllGatherFuncNet(nn.Cell):
    def construct(self, x):
        out, _ = all_gather_into_tensor(None, x)
        return out


class ReduceScatterFuncNet(nn.Cell):
    def construct(self, x):
        out, _ = reduce_scatter_tensor(None, x)
        return out


class AllGatherTransposeNet(nn.Cell):
    def construct(self, x):
        out, _ = all_gather_into_tensor(None, x)
        return out.transpose(1, 0)


class AllGatherSliceNet(nn.Cell):
    def construct(self, x):
        out, _ = all_gather_into_tensor(None, x)
        return out[::2]


class ReduceScatterTransposeNet(nn.Cell):
    def construct(self, x):
        out, _ = reduce_scatter_tensor(None, x)
        return out.transpose(1, 0)


class ReduceScatterSliceNet(nn.Cell):
    def construct(self, x):
        out, _ = reduce_scatter_tensor(None, x)
        return out[::2]


def test_hccl_all_gather_into_tensor_grad():
    """
    Feature: test 'all_gather_into_tensor' communication function in cell.
    Description: test 'all_gather_into_tensor' communication function in cell.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.ones([3, 4]).astype(np.float32))
    expect_output = np.ones([6, 4]).astype(np.float32)
    expect_output1 = np.ones([3, 4]).astype(np.float32) * 2
    net = AllGatherFuncNet()
    grad_fn = value_and_grad(net, grad_position=0)
    out, inputs_gradient = grad_fn(x)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)


def test_hccl_reduce_scatter_tensor_grad():
    """
    Feature: test 'reduce_scatter_tensor' communication function in cell.
    Description: test 'reduce_scatter_tensor' communication function in cell.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.ones([6, 4]).astype(np.float32)) * 2
    expect_output = np.ones([3, 4]).astype(np.float32) * 4
    expect_output1 = np.ones([6, 4]).astype(np.float32)
    net = ReduceScatterFuncNet()
    grad_fn = value_and_grad(net, grad_position=0)
    out, inputs_gradient = grad_fn(x)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)


def test_hccl_all_gather_into_tensor_grad_non_contiguous_transpose():
    """
    Feature: test 'all_gather_into_tensor' backward when upstream gradient is non-contiguous via transpose.
    Description: transpose after all_gather produces a non-contiguous gradient on backward,
                 which should be auto-contiguized before feeding into InnerCommAllGather.
    Expectation: expect correct result without raising 'not-contiguous tensor' exception.
    """
    x = ms.Tensor(np.ones([3, 4]).astype(np.float32))
    expect_output = np.ones([4, 6]).astype(np.float32)
    expect_output1 = np.ones([3, 4]).astype(np.float32) * 2
    net = AllGatherTransposeNet()
    grad_fn = value_and_grad(net, grad_position=0)
    out, inputs_gradient = grad_fn(x)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)


def test_hccl_all_gather_into_tensor_grad_non_contiguous_slice():
    """
    Feature: test 'all_gather_into_tensor' backward when upstream gradient is non-contiguous via strided slice.
    Description: strided slice after all_gather produces a non-contiguous gradient on backward,
                 which should be auto-contiguized before feeding into InnerCommAllGather.
    Expectation: expect correct result without raising 'not-contiguous tensor' exception.
    """
    x = ms.Tensor(np.ones([3, 4]).astype(np.float32))
    expect_output = np.ones([3, 4]).astype(np.float32)
    # slice[::2] backward yields non-contig dy[6,4] with rows 0,2,4 = 1s.
    # all_gather backward is reduce_scatter: sum across ranks then split per-rank.
    summed = np.zeros([6, 4]).astype(np.float32)
    summed[::2] = 2
    expect_output1 = summed[:3] if this_rank == 0 else summed[3:]
    net = AllGatherSliceNet()
    grad_fn = value_and_grad(net, grad_position=0)
    out, inputs_gradient = grad_fn(x)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)


def test_hccl_reduce_scatter_tensor_grad_non_contiguous_transpose():
    """
    Feature: test 'reduce_scatter_tensor' backward when upstream gradient is non-contiguous via transpose.
    Description: transpose after reduce_scatter produces a non-contiguous gradient on backward,
                 which should be auto-contiguized before feeding into InnerCommReduceScatter.
    Expectation: expect correct result without raising 'not-contiguous tensor' exception.
    """
    x = ms.Tensor(np.ones([6, 4]).astype(np.float32)) * 2
    expect_output = np.ones([4, 3]).astype(np.float32) * 4
    expect_output1 = np.ones([6, 4]).astype(np.float32)
    net = ReduceScatterTransposeNet()
    grad_fn = value_and_grad(net, grad_position=0)
    out, inputs_gradient = grad_fn(x)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)


def test_hccl_reduce_scatter_tensor_grad_non_contiguous_slice():
    """
    Feature: test 'reduce_scatter_tensor' backward when upstream gradient is non-contiguous via strided slice.
    Description: strided slice after reduce_scatter produces a non-contiguous gradient on backward,
                 which should be auto-contiguized before feeding into InnerCommReduceScatter.
    Expectation: expect correct result without raising 'not-contiguous tensor' exception.
    """
    x = ms.Tensor(np.ones([8, 4]).astype(np.float32)) * 2
    expect_output = np.ones([2, 4]).astype(np.float32) * 4
    expect_grad = np.zeros([4, 4]).astype(np.float32)
    expect_grad[::2] = 1
    expect_output1 = np.concatenate([expect_grad, expect_grad], axis=0)
    net = ReduceScatterSliceNet()
    grad_fn = value_and_grad(net, grad_position=0)
    out, inputs_gradient = grad_fn(x)
    assert np.allclose(out, expect_output)
    assert np.allclose(inputs_gradient, expect_output1)
