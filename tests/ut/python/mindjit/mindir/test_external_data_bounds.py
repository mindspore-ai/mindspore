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
