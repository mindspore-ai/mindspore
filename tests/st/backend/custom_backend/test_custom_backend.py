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
"""test custom backend"""
import os

import mindspore
from mindspore import jit, mint
import numpy as np
from tests.mark_utils import arg_mark


def get_ge_backend_path(file_name):
    os.system(f"python -c 'import mindspore;print(mindspore.__file__, flush=True)' > {file_name}.txt 2>&1")
    mindspore_path = os.popen(f"grep '__init__.py' {file_name}.txt").read()
    path = mindspore_path[:mindspore_path.rfind('/')] + "/lib/libmindspore_ge_backend.so"
    os.system(f"rm {file_name}.txt")
    return path

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01
          )
def test_custom_backend_build_and_registration():
    """
    Feature: test custom backend file existence and registration
    Description: verify plugin file properties and registration
    Expectation: verify builds successfully and registers successfully
    """
    ge_backend_path = get_ge_backend_path("test1")

    # Verufy ge_backend_path exist
    assert os.path.isfile(ge_backend_path), f"Plugin file {ge_backend_path} does not exist!"

    # Verify plugin file permissions
    assert os.access(ge_backend_path, os.R_OK), f"Plugin file is not readable: {ge_backend_path}"

    # Verify plugin file extension
    assert ge_backend_path.endswith('.so'), f"Plugin file does not have .so extension: {ge_backend_path}"

    # Verify plugin file registration
    success = mindspore.graph.register_custom_backend("ge", ge_backend_path)
    assert success, "Plugin registration failed"


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_custom_backend_run():
    """
    Feature: test custom backend build and run
    Description: verify custom backend build and run
    Expectation: verify custom backend build and run failed
    """
    ge_backend_path = get_ge_backend_path("test2")

    success = mindspore.graph.register_custom_backend("GE", ge_backend_path)
    assert success, "Plugin registration failed"

    x = mindspore.Tensor(np.ones([2, 2], np.float32))
    y = mindspore.Tensor(np.zeros([2, 2], np.float32))
    @jit(backend="GE")
    def net1(x, y):
        return mint.add(x, y)
    x = net1(x, y)


def test_different_backend_run():
    """
    Feature: test custom backend build and run
    Description: verify custom backend build and run
    Expectation: verify custom backend build and run failed
    """
    ge_backend_path = get_ge_backend_path("test3")

    success = mindspore.graph.register_custom_backend("GE", ge_backend_path)
    assert success, "Plugin registration failed"

    x = mindspore.Tensor(np.ones([2, 2], np.float32))
    y = mindspore.Tensor(np.zeros([2, 2], np.float32))

    @jit(backend="ms_backend")
    def net2(x):
        return mint.sin(x)
    x = net2(x)

    @jit
    def net3(x):
        return mint.cos(x)
    y = net3(y)

    @jit(backend="GE")
    def net4(x):
        return mint.cos(x)
    y = net4(y)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_different_custom_backend_run():
    """
    Feature: test custom backend build and run
    Description: verify custom backend build and run
    Expectation: verify custom backend build and run failed
    """
    os.system("GLOG_v=1 pytest -sv test_custom_backend.py::test_different_backend_run > log_custom_backend.txt 2>&1")
    ret = os.popen("grep -E 'Backend build graph|The created backend type' log_custom_backend.txt").read()
    assert "The created backend type: 1" in ret
    assert "Backend build graph, backend name: GE" in ret
    assert "The created backend type: 0" in ret
    os.system("rm log_custom_backend.txt")


def get_mindspore_root():
    """Get mindspore root path"""
    # Check environment variable
    if "MINDSPORE_ROOT" in os.environ:
        mindspore_root = os.environ["MINDSPORE_ROOT"]
        if os.path.exists(mindspore_root):
            return mindspore_root

    # Try to find through Python import
    try:
        import inspect
        mindspore_path = os.path.dirname(inspect.getfile(mindspore))

        # Check if include and lib directories exist in mindspore package
        include_path = os.path.join(mindspore_path, "include")
        lib_path = os.path.join(mindspore_path, "lib")

        if os.path.exists(include_path) and os.path.exists(lib_path):
            return mindspore_path
    except ImportError:
        pass
    return None


def test_real_custom_backend():
    """
    Feature: test custom backend build and run
    Description: verify custom backend build and run
    Expectation: verify custom backend build and run failed
    """
    custom_path = __file__[:__file__.rfind('/')] + "/resources/libcustom_backend.so"
    success = mindspore.graph.register_custom_backend("my_custom_backend", custom_path)
    assert success, "Plugin registration failed"

    x = mindspore.Tensor(np.ones([2, 2], np.float32))
    y = mindspore.Tensor(np.zeros([2, 2], np.float32))

    @jit(backend="my_custom_backend")
    def net1(x):
        return mint.sin(x)
    x = net1(x)

    @jit(backend="ms_backend")
    def net2(x):
        return mint.cos(x)
    y = net2(y)

@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_real_custom_backend_run():
    """
    Feature: test custom backend build and run
    Description: verify custom backend build and run
    Expectation: verify custom backend build and run failed
    """
    mindspore_root = get_mindspore_root()
    # build the plugin
    ret = os.system(f"cd resources; cmake . -DMINDSPORE_ROOT={mindspore_root}; make -j8")
    assert ret == 0
    os.path.isfile("resources/libcustom_backend.so")

    # load plugin and run
    os.system("GLOG_v=1 pytest -sv test_custom_backend.py::test_real_custom_backend > log_real_custom_backend.txt 2>&1")
    ret = os.popen("grep -E 'MSCustomBackendBase|Backend build graph' log_real_custom_backend.txt").read()
    assert "MSCustomBackendBase use the origin ms_backend to build the graph." in ret
    assert "MSCustomBackendBase use the origin ms_backend to run the graph." in ret
    # my_custom_backend use the ms_backend to build the graph.
    assert "Backend build graph, backend name: my_custom_backend" in ret
    assert "Backend build graph, backend name: ms_backend" in ret
    os.system("rm log_real_custom_backend.txt")
