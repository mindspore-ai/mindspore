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

"""test mindir external data path security"""

import os
import shutil
import sys

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, context
from mindspore.train.serialization import export, load
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model

context.set_context(mode=context.PYNATIVE_MODE)

_VAR_DIR_MAX_DEPTH = 10


def _export_mindir_with_external_data(prefix):
    """Export a split MindIR model whose weights are stored in external data files.

    The returned _graph.mindir has parameter_size > 0, each parameter carrying an
    external_data reference.  This exercises the ResolveExternalDataPath code path.
    """
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


def _export_parameterless_graph_mindir(prefix):
    """Create a _graph.mindir with zero parameters and a variables directory.

    A parameterless model exported with TOTAL_SAVE=0 goes through the
    save-together path, producing <prefix>.mindir.  We rename it to
    <prefix>_graph.mindir and create a <prefix>_variables directory so that
    the load code enters the get_all_files branch (triggered when
    parameter_size == 0 and the file ends with _graph.mindir).

    Returns (graph_file_path, variables_dir_path).
    """
    old_total_save = ms.train.serialization.TOTAL_SAVE
    ms.train.serialization.TOTAL_SAVE = 0
    try:
        class Net(nn.Cell):
            def construct(self, x):
                return x

        x = Tensor(np.array([1, 2, 3, 4], dtype=np.float32))
        export(Net(), x, file_name=prefix, file_format="MINDIR")
    finally:
        ms.train.serialization.TOTAL_SAVE = old_total_save

    original = prefix + ".mindir"
    graph_file = prefix + "_graph.mindir"
    os.rename(original, graph_file)

    var_dir = prefix + "_variables"
    if os.path.exists(var_dir):
        shutil.rmtree(var_dir)
    os.mkdir(var_dir)
    return graph_file, var_dir


def _tamper_external_data_location(graph_file, location):
    """Rewrite external_data.location in the MindIR protobuf."""
    model = mindir_model()
    with open(graph_file, "rb") as f:
        model.ParseFromString(f.read())
    params_with_external = [p for p in model.graph.parameter if p.HasField("external_data")]
    assert params_with_external, "Exported model has no external data parameters"
    param = params_with_external[0]
    param.external_data.location = location
    tampered = graph_file.replace(".mindir", "_tampered.mindir")
    with open(tampered, "wb") as f:
        f.write(model.SerializeToString())
    return tampered


@pytest.mark.parametrize("location", [
    "/etc/passwd",
    "../../etc/passwd",
    "../../../etc/passwd",
    "\\windows\\system32\\config\\sam",
    "data/../../../etc/passwd",
])
def test_load_mindir_rejects_traversal_location(tmp_path, location):
    """
    Feature: MindIR external data path validation.
    Description: A malicious external_data.location that is an absolute path or
                 contains parent traversal must be rejected.
    Expectation: RuntimeError is raised instead of reading arbitrary files.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("traversal_test")
        tampered_file = _tamper_external_data_location(graph_file, location)
        with pytest.raises(RuntimeError):
            load(tampered_file)
    finally:
        os.chdir(cwd)


def test_load_mindir_rejects_empty_location(tmp_path):
    """
    Feature: MindIR external data path validation .
    Description: An empty external_data.location must be rejected.
    Expectation: RuntimeError is raised.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("empty_loc_test")
        tampered_file = _tamper_external_data_location(graph_file, "")
        with pytest.raises(RuntimeError):
            load(tampered_file)
    finally:
        os.chdir(cwd)


def test_load_mindir_accepts_valid_external_data_location(tmp_path):
    """
    Feature: MindIR external data path validation.
    Description: A legitimately exported MindIR with external weights should load.
    Expectation: No exception.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file = _export_mindir_with_external_data("valid_loc_test")
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)


def test_load_mindir_completes_despite_deeply_nested_variables(tmp_path):
    """
    Feature: MindIR variables directory scan depth limit.
    Description: When the variables directory is nested deeper than the maximum
                 allowed depth (kVarDirMaxDepth=10), get_all_files stops scanning
                 deeper directories and returns false. The load succeeds with
                 whatever files were collected before hitting the limit.
    Expectation: Model loads successfully (empty variables directory, no crash).
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file, var_dir = _export_parameterless_graph_mindir("depth_test")
        deep_path = var_dir
        for i in range(_VAR_DIR_MAX_DEPTH):
            deep_path = os.path.join(deep_path, f"d{i}")
            os.mkdir(deep_path)
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)


def test_load_mindir_accepts_shallow_nested_variables(tmp_path):
    """
    Feature: MindIR variables directory scan depth limit.
    Description: A variables directory nested within the allowed depth should
                 not trigger the depth limit.
    Expectation: Model loads successfully (no depth-limit RuntimeError).
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file, var_dir = _export_parameterless_graph_mindir("shallow_test")
        deep_path = var_dir
        for i in range(_VAR_DIR_MAX_DEPTH - 1):
            deep_path = os.path.join(deep_path, f"d{i}")
            os.mkdir(deep_path)
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(sys.platform == "win32", reason="Symlink test requires Unix")
def test_load_mindir_skips_broken_symlink_in_variables(tmp_path):
    """
    Feature: MindIR variables directory symlink rejection.
    Description: A broken symbolic link in the variables directory must be
                 skipped.  Without the fix, stat() on the broken symlink would
                 fail and cause get_all_files to return false, making the load
                 fail.  With the fix, the symlink is skipped and the load
                 succeeds.
    Expectation: Model loads successfully (symlink was skipped).
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file, var_dir = _export_parameterless_graph_mindir("symlink_broken_test")
        symlink_path = os.path.join(var_dir, "evil_link")
        os.symlink("/nonexistent/path/should/not/exist", symlink_path)
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(sys.platform == "win32", reason="Symlink test requires Unix")
def test_load_mindir_skips_symlink_to_file_in_variables(tmp_path):
    """
    Feature: MindIR variables directory symlink rejection.
    Description: A symbolic link pointing to a file outside the variables
                 directory must be skipped to prevent escaping the model
                 directory.  Without the fix, the symlink target would be
                 scanned and parsed as a GraphProto, causing a load failure.
                 With the fix, the symlink is skipped and the load succeeds.
    Expectation: Model loads successfully (symlink was skipped).
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file, var_dir = _export_parameterless_graph_mindir("symlink_file_test")
        external_file = os.path.join(tmp_path, "external_secret")
        with open(external_file, "w", encoding="utf-8") as f:
            f.write("THIS IS NOT A VALID GRAPHPROTO")
        symlink_path = os.path.join(var_dir, "evil_link")
        os.symlink(external_file, symlink_path)
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(sys.platform == "win32", reason="Symlink test requires Unix")
def test_load_mindir_skips_symlink_to_directory_in_variables(tmp_path):
    """
    Feature: MindIR variables directory symlink rejection.
    Description: A symbolic link pointing to a directory outside the variables
                 directory must be skipped to prevent scanning arbitrary
                 directories.  Without the fix, get_all_files would recurse
                 into the symlinked directory and try to parse its files as
                 GraphProto, causing a load failure.  With the fix, the symlink
                 is skipped and the load succeeds.
    Expectation: Model loads successfully (symlink was skipped).
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file, var_dir = _export_parameterless_graph_mindir("symlink_dir_test")
        external_dir = os.path.join(tmp_path, "external_dir")
        os.mkdir(external_dir)
        decoy_file = os.path.join(external_dir, "decoy")
        with open(decoy_file, "w", encoding="utf-8") as f:
            f.write("DECOY DATA - NOT A VALID GRAPHPROTO")
        symlink_path = os.path.join(var_dir, "evil_dir_link")
        os.symlink(external_dir, symlink_path)
        graph = load(graph_file)
        assert graph is not None
    finally:
        os.chdir(cwd)


@pytest.mark.skipif(sys.platform == "win32", reason="Symlink test requires Unix")
def test_load_mindir_rejects_symlinked_variables_root(tmp_path):
    """
    Feature: MindIR variables directory symlink rejection.
    Description: If the variables directory itself is a symbolic link pointing
                 outside the model directory, the load must fail.  Without the
                 fix, get_all_files would follow the symlink and scan the target
                 directory, causing directory traversal.  With the fix, the root
                 symlink is rejected before scanning.
    Expectation: RuntimeError is raised instead of loading the model.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        graph_file, var_dir = _export_parameterless_graph_mindir("symlink_root_test")
        external_dir = os.path.join(tmp_path, "external_root_dir")
        os.mkdir(external_dir)
        decoy_file = os.path.join(external_dir, "decoy")
        with open(decoy_file, "w", encoding="utf-8") as f:
            f.write("DECOY DATA - NOT A VALID GRAPHPROTO")
        os.rmdir(var_dir)
        os.symlink(external_dir, var_dir)
        with pytest.raises(RuntimeError):
            load(graph_file)
    finally:
        os.chdir(cwd)
