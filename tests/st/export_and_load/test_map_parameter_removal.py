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
"""Protect ordinary parameter serialization when retiring MapParameter support."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn
from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.train.mind_ir_pb2 import ModelProto, AttributeProto, TensorProto
from tests.mark_utils import arg_mark


class ScaleNet(nn.Cell):
    """A normal Parameter with a deterministic forward result."""

    def __init__(self):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor([1.0, 2.0], ms.float32), name="weight")

    def construct(self, x):
        return x * self.weight


@pytest.fixture(autouse=True)
def restore_context_mode():
    """Keep the execution mode of neighboring serialization tests unchanged."""
    previous = ms.get_context("mode")
    yield
    ms.set_context(mode=previous)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('optimizer_type', [nn.Adam, nn.optim.FTRL, nn.optim.LazyAdam])
def test_normal_parameter_optimizer_checkpoint(mode, optimizer_type, tmp_path):
    """
    Feature: Ordinary optimizer updates and checkpoint roundtrip.
    Description: Update a normal Parameter and restore it after saving a Cell.
    Expectation: Dense optimizer branches and restored forward values are preserved.
    """
    ms.set_context(mode=mode)
    net = ScaleNet()
    optimizer = optimizer_type(net.trainable_params(), learning_rate=0.1)
    initial = net.weight.asnumpy().copy()
    gradient = np.array([0.2, -0.4], np.float32)
    optimizer((ms.Tensor(gradient),))
    saved = net.weight.asnumpy().copy()
    if optimizer_type is nn.optim.FTRL:
        root_accumulator = np.sqrt(0.1 + gradient ** 2)
        linear = gradient - (root_accumulator - np.sqrt(0.1)) / 0.1 * initial
        expected_weight = -linear * 0.1 / root_accumulator
    else:
        expected_weight = initial - 0.1 * gradient / (np.abs(gradient) + 1e-8 / np.sqrt(1 - 0.999))
    np.testing.assert_allclose(saved, expected_weight, rtol=1e-5, atol=1e-6)
    x = ms.Tensor([3.0, 4.0], ms.float32)
    expected = net(x).asnumpy()
    filename = str(tmp_path / "ordinary.ckpt")
    ms.save_checkpoint(net, filename, incremental=False)
    net.weight.set_data(ms.Tensor([0.0, 0.0], ms.float32))
    parameters = ms.load_checkpoint(filename)
    missing, unused = ms.load_param_into_net(net, parameters, strict_load=True)
    assert not missing and not unused
    np.testing.assert_array_equal(net.weight.asnumpy(), saved)
    np.testing.assert_allclose(net(x).asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_normal_parameter_mindir_roundtrip(tmp_path):
    """
    Feature: Ordinary parameter MindIR export and load.
    Description: Export a Cell with a normal Parameter and execute the loaded graph.
    Expectation: Removing MapParameter leaves normal parameter values intact.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    net = ScaleNet()
    x = ms.Tensor([3.0, 4.0], ms.float32)
    filename = str(tmp_path / "ordinary")
    ms.export(net, x, file_name=filename, file_format="MINDIR", incremental=False)
    loaded = nn.GraphCell(ms.load(filename + ".mindir"))
    np.testing.assert_allclose(loaded(x).asnumpy(), net(x).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('entrypoint', ['checkpoint', 'config', 'mindir'])
def test_incremental_map_request_rejected(entrypoint, tmp_path):
    """
    Feature: Removed incremental MapParameter interfaces.
    Description: Request incremental saving through the retained public signatures.
    Expectation: Unsupported requests fail explicitly, rather than silently saving dense data.
    """
    net = ScaleNet()
    with pytest.raises(ValueError, match="MapParameter has been removed"):
        if entrypoint == 'checkpoint':
            ms.save_checkpoint(net, str(tmp_path / "incremental.ckpt"), incremental=True)
        elif entrypoint == 'config':
            ms.train.CheckpointConfig(incremental=True)
        else:
            ms.export(net, ms.Tensor([3.0, 4.0], ms.float32), file_name=str(tmp_path / "incremental"),
                      file_format="MINDIR", incremental=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('include_dense', [False, True])
def test_legacy_map_checkpoint_rejected(include_dense, tmp_path):
    """
    Feature: Legacy MapTensor checkpoint compatibility.
    Description: Read a valid legacy wire field, alone or mixed with ordinary weights.
    Expectation: The loader rejects unsupported parameters without silently dropping them.
    """
    checkpoint = Checkpoint()
    if include_dense:
        dense = checkpoint.value.add()
        dense.tag = "weight"
        dense.tensor.dims.extend([2])
        dense.tensor.tensor_type = "Float32"
        dense.tensor.tensor_content = np.array([1.0, 2.0], np.float32).tobytes()
    legacy = checkpoint.value.add()
    legacy.tag = "legacy_table"
    for dtype, array in (("Int32", np.array([1], np.int32)),
                         ("Float32", np.array([[2.0]], np.float32)),
                         ("Int32", np.array([1], np.int32))):
        tensor = legacy.maptensor.tensor.add()
        tensor.dims.extend(array.shape)
        tensor.tensor_type = dtype
        tensor.tensor_content = array.tobytes()
    filename = tmp_path / "legacy.ckpt"
    filename.write_bytes(checkpoint.SerializeToString())
    with pytest.raises(ValueError, match="MapParameter has been removed"):
        ms.load_checkpoint(str(filename))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_legacy_map_mindir_rejected(tmp_path):
    """
    Feature: Legacy MapTensor MindIR compatibility.
    Description: Add a valid legacy map_parameter field to an ordinary exported graph.
    Expectation: Loading reports the unsupported feature before executing the graph.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    filename = str(tmp_path / "ordinary")
    ms.export(ScaleNet(), ms.Tensor([3.0, 4.0], ms.float32), file_name=filename, file_format="MINDIR")
    model = ModelProto()
    with open(filename + ".mindir", "rb") as file:
        model.ParseFromString(file.read())
    legacy = model.graph.map_parameter.add()
    legacy.name = "legacy_table"
    legacy.default_value.type = AttributeProto.STRING
    legacy.default_value.s = b"zeros"
    for tensor, dtype, array in ((legacy.key_tensor, TensorProto.INT32, np.array([1], np.int32)),
                                 (legacy.value_tensor, TensorProto.FLOAT, np.array([[2.0]], np.float32)),
                                 (legacy.status_tensor, TensorProto.INT32, np.array([1], np.int32))):
        tensor.data_type = dtype
        tensor.dims.extend(array.shape)
        tensor.raw_data = array.tobytes()
    legacy_file = tmp_path / "legacy.mindir"
    legacy_file.write_bytes(model.SerializeToString())
    with pytest.raises(RuntimeError, match="MapParameter has been removed"):
        ms.load(str(legacy_file))
