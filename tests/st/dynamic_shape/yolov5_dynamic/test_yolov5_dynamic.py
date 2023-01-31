# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import numpy as np
import pytest

from mindspore import log as logger
from tests.st.model_zoo_tests import utils


def init_files():
    cur_path = os.getcwd()
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "yolov5"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    # pylint: disable=anomalous-backslash-in-string
    old_list = ["gt_shape\[1\]"]
    # pylint: disable=anomalous-backslash-in-string
    new_list = ["-1"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/yolo.py"))
    os.system("""sed -i '1i\mode_name: "GRAPH"' {}""".format(os.path.join(cur_model_path, "default_config.yaml")))
    os.system("cp -r {} {}".format(os.path.join(cur_path, "run_yolov5_dynamic.py"), cur_model_path))
    return cur_model_path


def run_yolov5_dynamic_case(cur_model_path, device_target, mode="GRAPH"):
    exec_network_shell = "cd {}; python run_yolov5_dynamic.py --device_target={} --mode_name={} > log &".format(
        cur_model_path, device_target, mode)
    logger.warning("cmd [{}] is running...".format(exec_network_shell))
    os.system(exec_network_shell)
    cmd = "ps -ef | grep python | grep run_yolov5_dynamic.py | grep -v grep"
    ret = utils.process_check(100, cmd)
    if not ret:
        cmd = "{} | awk -F' ' '{{print $2}}' | xargs kill -9".format(cmd)
        os.system(cmd)
    assert ret
    log_file = os.path.join(cur_model_path, "log")
    loss_list = utils.get_loss_data_list(log_file)
    print("loss_list is: ", loss_list)
    assert len(loss_list) >= 3
    return loss_list


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov5_dynamic_gpu():
    """
    Feature: yolov5_dynamic
    Description: test yolov5_dynamic run
    Expectation: loss is same with the expect
    """
    cur_model_path = init_files()
    loss_list = run_yolov5_dynamic_case(cur_model_path, "GPU")
    expect_loss = [7200.505, 544.873, 600.88]
    # Different gpu device (such as V100 and 3090) lead to some differences
    # in the calculation results, so only the first 2 steps is compared
    assert np.allclose(loss_list[:2], expect_loss[:2], 1e-2, 1e-2)
    loss_list_pynative = run_yolov5_dynamic_case(cur_model_path, "GPU", "PYNATIVE")
    assert np.allclose(loss_list_pynative[:2], expect_loss[:2], 1e-2, 1e-2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_yolov5_dynamic_ascend():
    """
    Feature: yolov5_dynamic
    Description: test yolov5_dynamic run
    Expectation: loss is same with the expect
    """
    cur_model_path = init_files()
    loss_list = run_yolov5_dynamic_case(cur_model_path, "Ascend")
    expect_loss = [7200.35, 530]
    # Currently, the rtol/atol of loss of network running for many times exceeds
    # 1e-3, so only compare the first step
    assert np.allclose(loss_list[0], expect_loss[0], 1e-3, 1e-3)
