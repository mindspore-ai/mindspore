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
import pytest

from mindspore import context
from mindspore._c_expression import security
from tests.security_utils import security_off_wrap


# pylint: disable=W0212
# W0212: protected-access


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


def test_contex_create_context():
    """ test_contex_create_context """
    context.set_context(mode=context.PYNATIVE_MODE)
    if context.K_CONTEXT is None:
        ctx = context._context()
        assert ctx is not None
    context.K_CONTEXT = None


def test_set_save_graphs_in_security():
    """ test set save_graphs in the security mode"""
    if security.enable_security():
        with pytest.raises(ValueError) as err:
            context.set_context(save_graphs=True)
        assert "not supported" in str(err.value)


def test_set_save_graphs_path_in_security():
    """ test set save_graphs_path in the security mode"""
    if security.enable_security():
        with pytest.raises(ValueError) as err:
            context.set_context(save_graphs_path="ir_files")
        assert "not supported" in str(err.value)


def test_switch_mode():
    """ test_switch_mode """
    context.set_context(mode=context.GRAPH_MODE)
    assert context.get_context("mode") == context.GRAPH_MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    assert context.get_context("mode") == context.PYNATIVE_MODE


def test_set_device_id():
    """ test_set_device_id """
    with pytest.raises(TypeError):
        context.set_context(device_id=1)
        context.set_context(device_id="cpu")
    assert context.get_context("device_id") == 1


def test_device_target():
    """ test_device_target """
    with pytest.raises(TypeError):
        context.set_context(device_target=123)
    context.set_context(device_target="GPU")
    assert context.get_context("device_target") == "GPU"
    context.set_context(device_target="Ascend")
    assert context.get_context("device_target") == "Ascend"


def test_variable_memory_max_size():
    """test_variable_memory_max_size"""
    with pytest.raises(TypeError):
        context.set_context(variable_memory_max_size=True)
    with pytest.raises(TypeError):
        context.set_context(variable_memory_max_size=1)
        context.set_context(variable_memory_max_size="1G")
    context.set_context.__wrapped__(variable_memory_max_size="3GB")

def test_max_device_memory_size():
    """test_max_device_memory_size"""
    with pytest.raises(TypeError):
        context.set_context(max_device_memory=True)
    with pytest.raises(TypeError):
        context.set_context(max_device_memory=1)
        context.set_context(max_device_memory="3.5G")
    context.set_context.__wrapped__(max_device_memory="3GB")


def test_ascend_config():
    """"
    Feature: test_ascend_config function
    Description: Test case for simplest ascend_config
    Expectation: The results are as expected
    """
    context.set_context(device_target="Ascend")
    with pytest.raises(ValueError):
        context.set_context(precision_mode="force_fp16")
    with pytest.raises(ValueError):
        context.set_context(jit_compile=True)
    with pytest.raises(ValueError):
        context.set_context(ascend_config={"precision_mode": "xxx"})
    with pytest.raises(ValueError):
        context.set_context(ascend_config={"xxxx": 1})
    with pytest.raises(ValueError):
        context.set_context(ascend_config={"jit_compile": "xxx"})
    with pytest.raises(ValueError):
        context.set_context(ascend_config={"jit_compile": 2})
    with pytest.raises(ValueError):
        context.set_context(ascend_config={"precision_mode": 2})
    context.set_context.__wrapped__(ascend_config={"precision_mode": "force_fp16", "jit_compile": True})


def test_print_file_path():
    """test_print_file_path"""
    with pytest.raises(IOError):
        context.set_context(device_target="Ascend")
        context.set_context(print_file_path="./")


@security_off_wrap
def test_set_context():
    """ test_set_context """
    context.set_context.__wrapped__(device_id=0)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        save_graphs=True, save_graphs_path="mindspore_ir_path")
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
