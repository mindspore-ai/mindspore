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

from tests.st.model_zoo_tests import utils


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_FaceDetection_WIDER():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/research/cv".format(cur_path)
    model_name = "FaceDetection"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    old_list = ["max_epoch: 2500"]
    new_list = ["max_epoch: 1"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "default_config.yaml"))
    dataset_path = os.path.join(utils.data_root, "widerface/mindrecord_train/data.mindrecord")
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    model_train_command = "cd {}/scripts;sh run_standalone_train.sh Ascend {} {}"\
        .format(model_name, dataset_path, device_id)
    ret = os.system(model_train_command)
    assert ret == 0

    cmd = "ps -ef | grep train.py | grep Ascend | grep -v grep"
    ret = utils.process_check(150, cmd)
    assert ret

    log_file = os.path.join(cur_model_path, "scripts/device{}/train.log".format(device_id))
    pattern1 = r"loss\[([\d\.\+]+)\]"
    loss_list = utils.parse_log_file(pattern1, log_file)
    loss_list = loss_list[-10:]
    print("loss_list is", loss_list)
    assert sum(loss_list) / len(loss_list) < 12000
    pattern1 = r"\]\, ([\d\.\+]+) imgs\/sec"
    imgs_sec_list = utils.parse_log_file(pattern1, log_file)
    imgs_sec_list = imgs_sec_list[1:]
    print("imgs_sec_list is", imgs_sec_list)
    assert sum(imgs_sec_list) / len(imgs_sec_list) > 60
