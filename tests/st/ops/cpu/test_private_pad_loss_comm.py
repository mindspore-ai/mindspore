# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Regression tests for retired padding, loss and communication primitive exports."""
import inspect
from unittest.mock import Mock

import numpy as np
import pytest

from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import dtype as mstype
from mindspore.ops import operations
from mindspore.ops.operations import _inner_ops, comm_ops, nn_ops
from mindspore.ops._grad_experimental import grad_comm_ops
from mindspore.parallel._utils import _get_python_op
from mindspore.parallel.checkpoint_transform import _sync_params
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("module,name,args", [
    (nn_ops, "Pad", (((0, 0), (1, 1)),)), (comm_ops, "Send", (9, 1)),
    (comm_ops, "Receive", (9, 1, (2,), mstype.float32)),
    (nn_ops, "SoftmaxCrossEntropyWithLogits", ()), (nn_ops, "SparseSoftmaxCrossEntropyWithLogits", ()),
])
def test_private_pad_loss_comm_lookup(module, name, args):
    """
    Feature: Public padding, loss and communication primitive retirement.
    Description: Construct operators on the old and new internal lookup paths.
    Expectation: The public names are absent and the implementation classes are identical.
    """
    assert not hasattr(ops, name) and name not in ops.__all__
    assert not hasattr(operations, name) and name not in operations.__all__
    cls = getattr(module, name)
    assert getattr(_inner_ops, name) is cls
    for path in ("mindspore.ops.operations", "mindspore.ops.operations._inner_ops"):
        assert _get_python_op(name, path, 'lookup_test', list(args)).__class__ is cls


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("sparse", [False, True])
def test_retained_softmax_loss_and_gradient(sparse):
    """
    Feature: Retained nn cross entropy.
    Description: Compute mean loss and logits gradient for equivalent dense and sparse labels.
    Expectation: Values agree with a stable NumPy log-softmax reference.
    """
    logits = np.array([[1, 3, -2], [2, -1, 0]], np.float32)
    ids = np.array([1, 0], np.int32)
    one_hot = np.eye(3, dtype=np.float32)[ids]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    labels = Tensor(ids if sparse else one_hot)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction="mean")
    value, grad = ops.value_and_grad(loss)(Tensor(logits), labels)
    np.testing.assert_allclose(value.asnumpy(), -log_probs[np.arange(2), ids].mean(), rtol=1e-5)
    np.testing.assert_allclose(grad.asnumpy(), (np.exp(log_probs) - one_hot) / 2, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_retained_conv_output_padding():
    """
    Feature: Transposed convolution output padding.
    Description: Pad a strided transpose convolution with a unit kernel by one row and column.
    Expectation: Input values land at even indices and the extra border is zero.
    """
    layer = nn.Conv2dTranspose(1, 1, kernel_size=1, stride=2, pad_mode="pad", padding=0,
                               output_padding=(1, 1), weight_init="ones", has_bias=False)
    data = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    expected = np.zeros((1, 1, 4, 4), np.float32)
    expected[:, :, ::2, ::2] = data
    np.testing.assert_array_equal(layer(Tensor(data)).asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_communication_bprop_construction():
    """
    Feature: Communication bprop identity and metadata.
    Description: Construct registered bprop closures without executing point-to-point communication.
    Expectation: Send constructs Receive and vice versa, preserving the tag, rank and tensor shape.
    """
    send = comm_ops.Send(7, 1)
    send.add_prim_attr("shape", (2,))
    send.add_prim_attr("dtype", Tensor(np.float32(0)).dtype)
    getter = grad_comm_ops.bprop_getters.get(send, None)
    assert getter is grad_comm_ops.get_bprop_send
    recv = inspect.getclosurevars(getter(send)).nonlocals["send_grad"]
    assert recv.__class__ is comm_ops.Receive and recv.rank == 1 and recv.tag == 7
    reverse = grad_comm_ops.bprop_getters.get(recv, None)
    assert reverse is grad_comm_ops.get_bprop_receive
    back = inspect.getclosurevars(reverse(recv)).nonlocals["receive_grad"]
    assert back.__class__ is comm_ops.Send and back.rank == 1 and back.sr_tag == 7
    assert tuple(back.get_attr_dict()["shape"]) == (2,)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("is_send", [True, False])
def test_checkpoint_sync_dispatch(monkeypatch, is_send):
    """
    Feature: Pipeline checkpoint parameter synchronization.
    Description: Isolate transport and inspect the actual checkpoint dispatch and receive writeback.
    Expectation: Exactly one endpoint is called with the correct rank and tag.
    """
    param = Parameter(Tensor(np.array([1, 2], np.float32)), name="weight")
    received = Tensor(np.array([3, 4], np.float32))
    send, receive = Mock(), Mock()
    receive.return_value.return_value = received
    monkeypatch.setattr(_inner_ops, "Send", send)
    monkeypatch.setattr(_inner_ops, "Receive", receive)
    _sync_params("weight", param, [None] * 8 + [True, is_send, 2, 5])
    if is_send:
        send.assert_called_once_with(sr_tag=5, dest_rank=2)
        send.return_value.assert_called_once_with(param)
        receive.assert_not_called()
        np.testing.assert_array_equal(param.asnumpy(), [1, 2])
    else:
        receive.assert_called_once_with(sr_tag=5, src_rank=2, shape=param.shape, dtype=param.dtype)
        receive.return_value.assert_called_once_with(param)
        send.assert_not_called()
        np.testing.assert_array_equal(param.asnumpy(), received.asnumpy())
