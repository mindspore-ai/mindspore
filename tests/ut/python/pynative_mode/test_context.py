# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
""" test_context """
import os
import shutil
import json
import pytest

from mindspore import context


# pylint: disable=W0212
# W0212: protected-access


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


def test_contex_create_context():
    """ test_contex_create_context """
    context.set_context(mode=context.PYNATIVE_MODE)
    if context._k_context is None:
        ctx = context._context()
        assert ctx is not None
    context._k_context = None


def test_switch_mode():
    """ test_switch_mode """
    context.set_context(mode=context.GRAPH_MODE)
    assert context.get_context("mode") == context.GRAPH_MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    assert context.get_context("mode") == context.PYNATIVE_MODE


def test_set_device_id():
    """ test_set_device_id """
    with pytest.raises(TypeError):
        context.set_context(device_id="cpu")
    assert context.get_context("device_id") == 0
    context.set_context(device_id=1)
    assert context.get_context("device_id") == 1


def test_device_target():
    """ test_device_target """
    with pytest.raises(TypeError):
        context.set_context(device_target=123)
    context.set_context(device_target="GPU")
    assert context.get_context("device_target") == "GPU"
    context.set_context(device_target="Ascend")
    assert context.get_context("device_target") == "Ascend"
    assert context.get_context("device_id") == 1


def test_dump_target():
    """ test_dump_target """
    with pytest.raises(TypeError):
        context.set_context(save_dump_path=1)
    context.set_context(enable_dump=False)
    assert not context.get_context("enable_dump")
    context.set_context(enable_dump=True)
    assert context.get_context("enable_dump")
    assert context.get_context("save_dump_path") == "."


def test_enable_profiling():
    """ test_profiling_mode """
    with pytest.raises(TypeError):
        context.set_context(enable_profiling=1)
    with pytest.raises(TypeError):
        context.set_context(enable_profiling="1")
    context.set_context(enable_profiling=True)
    assert context.get_context("enable_profiling") is True
    context.set_context(enable_profiling=False)
    assert context.get_context("enable_profiling") is False


def test_profiling_options():
    """ test_profiling_options """
    with pytest.raises(TypeError):
        context.set_context(profiling_options=True)
    with pytest.raises(TypeError):
        context.set_context(profiling_options=1)
    profiling_options = {
        "output": "",
        "fp_point": "",
        "bp_point": "",
        "training_trace": "on",
        "task_trace": "on",
        "aic_metrics": "PipeUtilization",
        "aicpu": "on"
    }
    profiling_options = json.dumps(profiling_options)
    context.set_context(profiling_options=profiling_options)
    assert context.get_context("profiling_options") == profiling_options


def test_variable_memory_max_size():
    """test_variable_memory_max_size"""
    with pytest.raises(TypeError):
        context.set_context(variable_memory_max_size=True)
    with pytest.raises(TypeError):
        context.set_context(variable_memory_max_size=1)
    with pytest.raises(ValueError):
        context.set_context(variable_memory_max_size="")
    with pytest.raises(ValueError):
        context.set_context(variable_memory_max_size="1G")
    with pytest.raises(ValueError):
        context.set_context(variable_memory_max_size="32GB")
    context.set_context(variable_memory_max_size="3GB")


def test_print_file_path():
    """test_print_file_path"""
    with pytest.raises(IOError):
        context.set_context(print_file_path="./")


def test_set_context():
    """ test_set_context """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        device_id=0, save_graphs=True, save_graphs_path="mindspore_ir_path")
    assert context.get_context("device_id") == 0
    assert context.get_context("device_target") == "Ascend"
    assert context.get_context("save_graphs")
    assert os.path.exists("mindspore_ir_path")
    assert context.get_context("save_graphs_path").find("mindspore_ir_path") > 0
    assert context.get_context("mode") == context.GRAPH_MODE

    context.set_context(mode=context.PYNATIVE_MODE)
    assert context.get_context("mode") == context.PYNATIVE_MODE
    assert context.get_context("device_target") == "Ascend"

    with pytest.raises(ValueError):
        context.set_context(modex="ge")


def teardown_module():
    dirs = ['mindspore_ir_path']
    for item in dirs:
        item_name = './' + item
        if not os.path.exists(item_name):
            continue
        if os.path.isdir(item_name):
            shutil.rmtree(item_name)
        elif os.path.isfile(item_name):
            os.remove(item_name)
