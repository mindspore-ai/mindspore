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
"""Test external data offset/length bounds validation when loading MindIR."""
import os

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, context
from mindspore.train.serialization import export, load
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model
from mindspore.train.mind_ir_pb2 import TensorProto as mindir_tensor

context.set_context(mode=context.PYNATIVE_MODE)


def _export_mindir_with_external_data(prefix):
    """Export a MindIR model whose weights are stored in external data files."""
    old_total_save = ms.train.serialization.TOTAL_SAVE
    ms.train.serialization.TOTAL_SAVE = 0
    try:
        class Net(nn.Cell):
            def __init__(self):
                super().__init__()
                self.y = Parameter(Tensor(np.array([2, 3, 3, 4], dtype=np.float32)), name="w")
                self.add = ops.Add()

            def construct(self, x):
                return self.add(x, self.y)

        x = Tensor(np.array([1, 2, 3, 4], dtype=np.float32))
        export(Net(), x, file_name=prefix, file_format="MINDIR")
    finally:
        ms.train.serialization.TOTAL_SAVE = old_total_save
    return prefix + "_graph.mindir"


def _tamper_tensor_shape(graph_file, dims):
    """Rewrite the first external parameter shape in the MindIR protobuf."""
    model = mindir_model()
    with open(graph_file, "rb") as f:
        model.ParseFromString(f.read())
    params_with_external = [p for p in model.graph.parameter if p.HasField("external_data")]
    assert params_with_external, "Exported model has no external data parameters"
    param = params_with_external[0]
    del param.dims[:]
    param.dims.extend(dims)
    tampered = graph_file.replace(".mindir", "_shape_tampered.mindir")
    with open(tampered, "wb") as f:
        f.write(model.SerializeToString())
    return tampered


def _tamper_tensor_data_type(graph_file, data_type):
    """Rewrite the first external parameter data type in the MindIR protobuf."""
    model = mindir_model()
    with open(graph_file, "rb") as f:
        model.ParseFromString(f.read())
    params_with_external = [p for p in model.graph.parameter if p.HasField("external_data")]
    assert params_with_external, "Exported model has no external data parameters"
    params_with_external[0].data_type = data_type
    tampered = graph_file.replace(".mindir", "_type_tampered.mindir")
    with open(tampered, "wb") as f:
        f.write(model.SerializeToString())
    return tampered


def _tamper_external_data(graph_file, offset, length):
    """Rewrite external_data offset/length fields in the MindIR protobuf."""
    model = mindir_model()
    with open(graph_file, "rb") as f:
        model.ParseFromString(f.read())
    params_with_external = [p for p in model.graph.parameter if p.HasField("external_data")]
    assert params_with_external, "Exported model has no external data parameters"
    param = params_with_external[0]
    param.external_data.offset = offset
    param.external_data.length = length
    tampered = graph_file.replace(".mindir", "_tampered.mindir")
    with open(tampered, "wb") as f:
        f.write(model.SerializeToString())
    return tampered


@pytest.mark.parametrize("offset,length", [
    (-100, 16),
    (-1, 16),
    (-1, 1),   # offset+length==0 bypasses old joint check, but data+offset is OOB
    (64, -1),
    (64, 10 ** 9),
])
def test_load_mindir_rejects_invalid_external_data_bounds(tmp_path, offset, length):
    """
    Feature: MindIR external data bounds validation.
    Description: Loading a tampered MindIR with invalid external_data offset/length must fail.
    Expectation: RuntimeError is raised instead of performing out-of-bounds reads.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("external_bounds")
        tampered_file = _tamper_external_data(graph_file, offset, length)
        with pytest.raises(RuntimeError, match="Load MindIR failed"):
            load(tampered_file)
    finally:
        os.chdir(cwd)


def test_load_mindir_accepts_valid_external_data_bounds(tmp_path):
    """
    Feature: MindIR external data bounds validation.
    Description: A legitimately exported MindIR with external weights should still load.
    Expectation: No exception.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("external_bounds_valid")
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("dims", [
    [-1],
    [2 ** 63 - 1],
    [2 ** 31 - 1] * 3,  # each dimension is valid but their product exceeds the byte budget
    [300_000_000],  # declared shape is inconsistent with the short external data
    [1] * 33,
])
def test_load_mindir_rejects_unsafe_tensor_shapes_before_allocation(tmp_path, dims):
    """
    Feature: MindIR TensorProto resource limits.
    Description: Invalid, overflowing, or source-inconsistent shapes are rejected before allocating tensor memory.
    Expectation: RuntimeError is raised without attempting a large allocation.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("tensor_shape_limits")
        tampered_file = _tamper_tensor_shape(graph_file, dims)
        with pytest.raises(RuntimeError, match="Load MindIR failed"):
            load(tampered_file)
    finally:
        os.chdir(cwd)


def test_load_mindir_rejects_external_length_shape_mismatch(tmp_path):
    """
    Feature: MindIR TensorProto data length validation.
    Description: Load an uncompressed external tensor whose declared length mismatches its shape-derived byte size.
    Expectation: RuntimeError is raised instead of loading a partially initialized tensor.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("tensor_length_mismatch")
        tampered_file = _tamper_external_data(graph_file, 1, 8)
        with pytest.raises(RuntimeError, match="Load MindIR failed"):
            load(tampered_file)
    finally:
        os.chdir(cwd)


def test_load_mindir_rejects_unsupported_string_tensor_layout(tmp_path):
    """
    Feature: MindIR STRING Tensor data layout validation.
    Description: Load a STRING Tensor whose serialized length cannot be inferred from its shape.
    Expectation: RuntimeError is raised before allocating or copying an incorrectly sized buffer.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("string_tensor_layout")
        tampered_file = _tamper_tensor_data_type(graph_file, mindir_tensor.STRING)
        with pytest.raises(RuntimeError, match="Load MindIR failed"):
            load(tampered_file)
    finally:
        os.chdir(cwd)


def test_load_mindir_reads_only_declared_external_segment(tmp_path):
    """
    Feature: MindIR external data streaming.
    Description: Load a small Tensor from an external file with unrelated trailing data.
    Expectation: The model loads without buffering the complete external file.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("external_segment")
        model = mindir_model()
        with open(graph_file, "rb") as f:
            model.ParseFromString(f.read())
        params_with_external = [p for p in model.graph.parameter if p.HasField("external_data")]
        assert params_with_external, "Exported model has no external data parameters"
        external_data_file = params_with_external[0].external_data.location
        # MindIR export intentionally makes external data read-only; make the fixture writable before tampering.
        os.chmod(external_data_file, 0o600)
        with open(external_data_file, "ab") as f:
            f.write(bytes(2 * 1024 * 1024))
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)
