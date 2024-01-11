# Copyright 2021 Huawei Technologies Co., Ltd
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
import pytest

from mindspore import log as logger
from tests.st.model_zoo_tests import utils

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_resnet50_cifar10_gpu_accuracy():
    """
    Feature: None
    Description: test resnet50_cifar10_gpu accuracy
    Expectation: None
    """
    cur_path = os.getcwd()
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "resnet"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, "resnet")
    old_list = ["total_epochs=config.epoch_size", "config.epoch_size - config.pretrain_epoch_size"]
    new_list = ["total_epochs=10", "10"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "train.py"))
    old_list = ["from mindspore._checkparam import Validator"]
    new_list = ["from mindspore import _checkparam as Validator"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/momentum.py"))
    dataset_path = os.path.join(utils.data_root, "cifar-10-batches-bin")
    config_path = os.path.join(cur_model_path, "config", "resnet50_cifar10_config.yaml")
    os.system("nvidia-smi")
    os.system("fuser -v /dev/nvidia*")
    exec_network_shell = "cd resnet/scripts; sh run_distribute_train_gpu.sh {} {}" \
        .format(dataset_path, config_path)
    logger.warning("cmd [{}] is running...".format(exec_network_shell))
    os.system(exec_network_shell)
    cmd = "ps -ef | grep python | grep train.py | grep -v grep"
    ret = utils.process_check(100, cmd)
    if not ret:
        cmd = "{} | awk -F' ' '{{print $2}}' | xargs kill -9".format(cmd)
        os.system(cmd)
    assert ret
    log_file = os.path.join(cur_model_path, "scripts/train_parallel/log")
    loss_list = utils.get_loss_data_list(log_file)[-8:]
    print("loss_list is", loss_list)
    assert sum(loss_list) / len(loss_list) < 0.70

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet50_cifar10_gpu_performance():
    """
    Feature: None
    Description: test resnet50_cifar10_gpu performance
    Expectation: None
    """
    cur_path = os.getcwd()
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "resnet"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, "resnet")
    old_list = ["from mindspore._checkparam import Validator"]
    new_list = ["from mindspore import _checkparam as Validator"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/momentum.py"))
    os.system("cp -r {} {}".format(os.path.join(cur_path, "train.py"), cur_model_path))
    config_path = os.path.join(cur_model_path, "config", "resnet50_cifar10_config.yaml")
    os.system("nvidia-smi")
    os.system("fuser -v /dev/nvidia*")
    exec_network_shell = "cd {}; python train.py --device_target=GPU --output_path=./output " \
                         "--config_path={} &> log &".format(cur_model_path, config_path)
    logger.warning("cmd [{}] is running...".format(exec_network_shell))
    os.system(exec_network_shell)
    cmd = "ps -ef | grep python | grep train.py | grep -v grep"
    ret = utils.process_check(100, cmd)
    if not ret:
        cmd = "{} | awk -F' ' '{{print $2}}' | xargs kill -9".format(cmd)
        os.system(cmd)
    assert ret
    log_file = os.path.join(cur_model_path, "log")
    pattern = r"cost time: ([\d\.]+) ms"
    step_time_list = utils.parse_log_file(pattern, log_file)[1:]
    per_step_time = sum(step_time_list) / len(step_time_list)
    print("per_step_time is", per_step_time)
    assert per_step_time < 70
