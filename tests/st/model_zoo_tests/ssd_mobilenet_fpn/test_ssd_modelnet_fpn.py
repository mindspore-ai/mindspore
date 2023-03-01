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
@pytest.mark.env_single
def test_SSD_mobilenet_v1_fpn_coco2017():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "ssd"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    os.system("cp -r {} {}".format(os.path.join(cur_path, "train.py"), cur_model_path))

    old_list = ["/cache/data", "MindRecord_COCO", "coco_ori", "/ckpt/mobilenet_v1.ckpt"]
    new_list = [os.path.join(utils.data_root, "coco/coco2017"), "mindrecord_train/ssd_mindrecord", ".",
                os.path.join(utils.ckpt_root, "ssd_mobilenet_v1/mobilenet-v1.ckpt")]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "config/ssd_mobilenet_v1_fpn_config.yaml"))

    exec_network_shell = "cd {0}; sh -x scripts/run_distribute_train.sh 8 {1} 0.2 coco \
        {2} config/ssd_mobilenet_v1_fpn_config.yaml".format(model_name, 60, utils.rank_table_path)
    os.system(exec_network_shell)
    cmd = "ps -ef --columns 1000 | grep train.py | grep coco | grep device_num | grep device_id | grep -v grep"
    ret = utils.process_check(120, cmd)
    assert ret

    log_file = os.path.join(cur_model_path, "LOG{}/log.txt")
    for i in range(8):
        per_step_time = utils.get_perf_data(log_file.format(i))
        print("per_step_time is", per_step_time)
        assert per_step_time < 580
    loss_list = []
    for i in range(8):
        loss = utils.get_loss_data_list(log_file.format(i))
        print("loss is", loss[-1])
        loss_list.append(loss[-1])
    assert 500000 < sum(loss_list) / len(loss_list) < 600000
