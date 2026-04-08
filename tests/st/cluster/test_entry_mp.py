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
"""
This test is for multiprocessing launch.
"""
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_mp_init_method():
    """
    Feature: initialize cluster with multiprocessing module.
    Description: initialize cluster with multiprocessing module.
    Expectation: All workers are successfully spawned and running communication op.
    """
    env = "export INIT_DEFAULT_GROUP=True;"
    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_mp_init_method.py"
    assert os.path.exists(script)

    cmd = f"{env} python {script}"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_mp_init_process_group_with_group_name():
    """
    Feature: initialize cluster with multiprocessing module, set group_name for init_process_group.
    Description: initialize cluster with multiprocessing module, set group_name for init_process_group.
    Expectation: All workers are successfully spawned and running communication op.
    """
    env = "export INIT_CROSS_DEFAULT_GROUP=True;"
    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_mp_init_method.py"
    assert os.path.exists(script)

    cmd = f"{env} python {script}"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_mp_tcpstore():
    """
    Feature: initialize cluster with multiprocessing module.
    Description: initialize cluster with multiprocessing module.
    Expectation: All workers are successfully spawned and running communication op.
    """
    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_mp_tcpstore.py"
    assert os.path.exists(script)

    cmd = f"python {script}"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_mp_init_method_cross_default_group():
    """
    Feature: initialize cluster with multiprocessing module, support multi-init with init_method
        and cross default group.
    Description: initialize cluster with multiprocessing module.
    Expectation: All workers are successfully spawned and running communication op.
    """
    env = "export INIT_DEFAULT_GROUP=True;"
    env += "export INIT_CROSS_DEFAULT_GROUP=True;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_mp_init_method.py"
    assert os.path.exists(script)

    cmd = f"{env} python {script}"
    return_code = os.system(cmd)
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_mp_tcpstore_cross_default_group():
    """
    Feature: initialize cluster with multiprocessing module, support multi-init with tcpstore and cross default group.
    Description: initialize cluster with multiprocessing module.
    Expectation: All workers are successfully spawned and running communication op.
    """
    env = "export INIT_CROSS_DEFAULT_GROUP=True;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_mp_tcpstore.py"
    assert os.path.exists(script)

    cmd = f"{env} python {script}"
    return_code = os.system(cmd)
    assert return_code == 0
