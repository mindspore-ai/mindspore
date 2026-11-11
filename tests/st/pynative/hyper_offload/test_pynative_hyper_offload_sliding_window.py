# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Simple tests for sliding_window offload/reload in PyNative hyper_offload.

Coverage: ordinary op, single view, multi view, view-of-view chain, and inplace op.
"""

import os

import numpy as np
from mindspore.ops.operations import _inplace_ops
import mindspore as ms
from mindspore import Tensor, context, nn, ops
from mindspore._c_expression import OffloadManager

RTOL = 1e-4
ATOL = 1e-4
TENSOR_SHAPE = (160, 65536)


# ==============================================================================
# Networks
# ==============================================================================

class SimpleOrdinaryOpNet(nn.Cell):
    """Minimal network: one ordinary-op source, pressure ops, then reuse the source."""

    def construct(self, x):
        source = x + 1.0
        x = x * 0.5
        x = x + 2.0
        return x + source


class SimpleViewOpNet(nn.Cell):
    """Minimal network: view alias of a source tensor, reused after pressure ops.

    Uses a 3-D reshape that preserves the C-contiguous memory layout so the
    view is a true zero-copy alias rather than an implicit data copy.
    """

    def construct(self, x):
        source = x + 1.0
        # (160, 65536) -> (160, 256, 256): contiguous, zero-copy view
        view = source.reshape(TENSOR_SHAPE[0], 256, 256)
        x = x * 0.5
        x = x + 2.0
        # Serialize additions to keep peak live tensors low
        x = x + source
        x = x + view.reshape(TENSOR_SHAPE[0], TENSOR_SHAPE[1])
        return x


class SimpleMultiViewOpNet(nn.Cell):
    """Network with two independent source tensors, each owning a view alias.

    All reshapes use contiguous layouts so every view is a zero-copy alias.
    """

    def construct(self, x):
        source1 = x + 1.0
        view1 = source1.reshape(TENSOR_SHAPE[0], 256, 256)
        source2 = x * 0.5
        view2 = source2.reshape(TENSOR_SHAPE[0], 256, 256)
        x = x * 0.5
        x = x + 2.0
        x = x + source1 + view1.reshape(TENSOR_SHAPE[0], TENSOR_SHAPE[1])
        x = x + source2 + view2.reshape(TENSOR_SHAPE[0], TENSOR_SHAPE[1])
        return x


class SimpleViewOfViewOpNet(nn.Cell):
    """Network with a view-of-view chain: 2D → 3D → 4D → 2D.

    Every reshape in the chain preserves C-contiguous layout so each
    intermediate is a true zero-copy alias of the original storage.
    """

    def construct(self, x):
        source = x + 1.0
        # (160, 65536) → (160, 256, 256) → (160, 16, 16, 256) → (160, 65536)
        view_3d = source.reshape(TENSOR_SHAPE[0], 256, 256)
        view_4d = view_3d.reshape(TENSOR_SHAPE[0], 16, 16, 256)
        view_2d = view_4d.reshape(TENSOR_SHAPE[0], TENSOR_SHAPE[1])
        x = x * 0.5
        x = x + 2.0
        x = x + source + view_2d
        return x


class SimpleInplaceOpNet(nn.Cell):
    """Minimal network: inplace-modified source tensor, reused after pressure ops."""

    def __init__(self):
        super().__init__()
        self.inplace_add = _inplace_ops.InplaceAdd((0,))
        self.update = Tensor(np.full((1, TENSOR_SHAPE[1]), 0.25, np.float32))

    def construct(self, x):
        source = x + 1.0
        source = self.inplace_add(source, self.update)
        x = x * 0.5
        x = x + 2.0
        return x + source


# ==============================================================================
# Helpers
# ==============================================================================

def _run_forward(net, x_np):
    """Run network forward and return numpy output."""
    x = Tensor(x_np)
    output = net(x)
    return output.asnumpy()


def _run_forward_and_grad(net, x_np):
    """Run network forward and input gradient."""
    x = Tensor(x_np)
    output = net(x)
    output_np = output.asnumpy()
    del output

    def forward_fn(inp):
        return ops.ReduceSum()(net(inp))

    grad = ms.grad(forward_fn)(x)
    return output_np, grad.asnumpy()


def _expect_ordinary_output_and_grad(x_np):
    """Expected values for SimpleOrdinaryOpNet.

    source = x + 1
    x' = 0.5*x + 2
    output = x' + source = (0.5*x + 2) + (x + 1) = 1.5*x + 3
    d(output)/dx = 1.5
    """
    output = 1.5 * x_np + 3.0
    grad = np.full_like(x_np, 1.5)
    return output.astype(np.float32), grad.astype(np.float32)


def _expect_view_output_and_grad(x_np):
    """Expected values for SimpleViewOpNet.

    source = x + 1
    view = reshape(source)          # alias, same data as source
    x' = 0.5*x + 2
    output = x' + source + view_back = (0.5*x+2) + (x+1) + (x+1) = 2.5*x + 4
    d(output)/dx = 2.5
    """
    output = 2.5 * x_np + 4.0
    grad = np.full_like(x_np, 2.5)
    return output.astype(np.float32), grad.astype(np.float32)


def _expect_inplace_output(x_np):
    """Expected output for SimpleInplaceOpNet.

    source = x + 1
    source[0, :] += 0.25             # inplace update on row 0
    x' = 0.5*x + 2
    output = x' + source
      output[0, :]   = 1.5*x[0,:] + 3.25
      output[1:, :]  = 1.5*x[1:,:] + 3.0
    """
    source = x_np + 1.0
    source[0, :] += 0.25
    output = 0.5 * x_np + 2.0 + source
    return output.astype(np.float32)


def _expect_multi_view_output_and_grad(x_np):
    """Expected values for SimpleMultiViewOpNet.

    source1 = x + 1        view1 = reshape(source1)   # alias
    source2 = 0.5*x        view2 = reshape(source2)   # alias
    x' = 0.5*x + 2
    output = x' + source1 + view1_back + source2 + view2_back
           = (0.5*x+2) + (x+1) + (x+1) + (0.5*x) + (0.5*x)
           = 3.5*x + 4
    d(output)/dx = 3.5
    """
    output = 3.5 * x_np + 4.0
    grad = np.full_like(x_np, 3.5)
    return output.astype(np.float32), grad.astype(np.float32)


def _expect_view_of_view_output_and_grad(x_np):
    """Expected values for SimpleViewOfViewOpNet.

    source = x + 1
    Every reshape in the chain is contiguous → zero-copy alias.
    view_2d = reshape(reshape(reshape(source))) = source = x + 1
    x' = 0.5*x + 2
    output = x' + source + view_2d = (0.5*x+2) + (x+1) + (x+1) = 2.5*x + 4
    d(output)/dx = 2.5
    """
    output = 2.5 * x_np + 4.0
    grad = np.full_like(x_np, 2.5)
    return output.astype(np.float32), grad.astype(np.float32)


def _make_input():
    """Create one ~40MB input tensor as numpy data."""
    np.random.seed(0)
    return np.random.randn(*TENSOR_SHAPE).astype(np.float32)


# ==============================================================================
# Test Class
# ==============================================================================

class TestHyperOffloadSlidingWindowSimple:
    """Test PyNative hyper_offload sliding_window with minimal ordinary, view, and inplace networks."""

    @classmethod
    def setup_class(cls):
        """Set device memory budget once for the whole test class."""
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    def setup_method(self):
        """Set up test environment."""
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
        self.origin_hyper_offload = os.environ.get("MS_DEV_HYPER_OFFLOAD")
        self.origin_sliding_window_buffer = os.environ.get("MS_DEV_SLIDING_WINDOW_BUFFER_SIZE_MB")
        os.environ["MS_DEV_HYPER_OFFLOAD"] = "1"
        os.environ["MS_DEV_SLIDING_WINDOW_BUFFER_SIZE_MB"] = "48"
        OffloadManager.get_instance().reset()

    def teardown_method(self):
        """Clean up after each test."""
        if self.origin_hyper_offload is None:
            os.environ.pop("MS_DEV_HYPER_OFFLOAD", None)
        else:
            os.environ["MS_DEV_HYPER_OFFLOAD"] = self.origin_hyper_offload
        if self.origin_sliding_window_buffer is None:
            os.environ.pop("MS_DEV_SLIDING_WINDOW_BUFFER_SIZE_MB", None)
        else:
            os.environ["MS_DEV_SLIDING_WINDOW_BUFFER_SIZE_MB"] = self.origin_sliding_window_buffer
        OffloadManager.get_instance().reset()

    def test_sliding_window_offload_reload_ordinary(self):
        """
        Feature: PyNative hyper_offload sliding_window offload and reload.
        Description: A single ordinary-op source is reused after pressure ops.
        Expectation: Output and gradient should match the numpy reference.
        """
        x_np = _make_input()
        net = SimpleOrdinaryOpNet()
        net.auto_offload()
        net.set_grad()
        output, grad = _run_forward_and_grad(net, x_np)
        expect_output, expect_grad = _expect_ordinary_output_and_grad(x_np)
        np.testing.assert_allclose(output, expect_output, rtol=RTOL, atol=ATOL, err_msg="ordinary: output mismatch")
        np.testing.assert_allclose(grad, expect_grad, rtol=RTOL, atol=ATOL, err_msg="ordinary: gradient mismatch")

    def test_sliding_window_offload_reload_view(self):
        """
        Feature: PyNative hyper_offload sliding_window offload and reload.
        Description: A view alias of a source is created and reused together with the source after pressure ops.
        Expectation: Output and gradient should match the numpy reference.
        """
        x_np = _make_input()
        net = SimpleViewOpNet()
        net.auto_offload()
        net.set_grad()
        output, grad = _run_forward_and_grad(net, x_np)
        expect_output, expect_grad = _expect_view_output_and_grad(x_np)
        np.testing.assert_allclose(output, expect_output, rtol=RTOL, atol=ATOL, err_msg="view: output mismatch")
        np.testing.assert_allclose(grad, expect_grad, rtol=RTOL, atol=ATOL, err_msg="view: gradient mismatch")

    def test_sliding_window_offload_reload_inplace(self):
        """
        Feature: PyNative hyper_offload sliding_window offload and reload.
        Description: An inplace-modified source is reused after pressure ops.
        Expectation: Output should match the numpy reference (gradient not checked for inplace ops).
        """
        x_np = _make_input()
        net = SimpleInplaceOpNet()
        net.auto_offload()
        # Note: InplaceAdd may not have full gradient support, so forward-only check.
        output = _run_forward(net, x_np)
        expect_output = _expect_inplace_output(x_np)
        np.testing.assert_allclose(output, expect_output, rtol=RTOL, atol=ATOL, err_msg="inplace: output mismatch")

    def test_sliding_window_offload_reload_multi_view(self):
        """
        Feature: PyNative hyper_offload sliding_window offload and reload.
        Description: Two independent source tensors, each with a view alias, are reused after pressure ops.
        Expectation: Output and gradient should match the numpy reference.
        """
        x_np = _make_input()
        net = SimpleMultiViewOpNet()
        net.auto_offload()
        net.set_grad()
        output, grad = _run_forward_and_grad(net, x_np)
        expect_output, expect_grad = _expect_multi_view_output_and_grad(x_np)
        np.testing.assert_allclose(output, expect_output, rtol=RTOL, atol=ATOL, err_msg="multi_view: output mismatch")
        np.testing.assert_allclose(grad, expect_grad, rtol=RTOL, atol=ATOL, err_msg="multi_view: gradient mismatch")

    def test_sliding_window_offload_reload_view_of_view(self):
        """
        Feature: PyNative hyper_offload sliding_window offload and reload.
        Description: A view-of-view chain (2D→3D→4D→2D) is created and reused after pressure ops.
        Expectation: Output and gradient should match the numpy reference.
        """
        x_np = _make_input()
        net = SimpleViewOfViewOpNet()
        net.auto_offload()
        net.set_grad()
        output, grad = _run_forward_and_grad(net, x_np)
        expect_output, expect_grad = _expect_view_of_view_output_and_grad(x_np)
        np.testing.assert_allclose(output, expect_output, rtol=RTOL, atol=ATOL, err_msg="view_of_view: output mismatch")
        np.testing.assert_allclose(grad, expect_grad, rtol=RTOL, atol=ATOL, err_msg="view_of_view: gradient mismatch")
