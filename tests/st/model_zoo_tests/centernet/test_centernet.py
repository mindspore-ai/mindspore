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
def test_center_net():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/research/cv".format(cur_path)
    model_name = "centernet"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    old_list = ['new_repeat_count, dataset', 'args_opt.data_sink_steps']
    new_list = ['5, dataset', '20']
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "train.py"))
    old_list = ["device_ips = {}", "device_ip.strip()",
                "rank_size = 0", "this_server = server",
                "this_server\\[\\\"device\\\"\\]",
                "instance\\[\\\"device_id\\\"\\]"]
    new_list = ["device_ips = {}\\n    '''", "device_ip.strip()\\n    '''",
                "rank_size = 8\\n    this_server = hccl_config[\\\"group_list\\\"][0]\\n    '''",
                "this_server = server\\n    '''",
                "this_server[\\\"instance_list\\\"]",
                "instance[\\\"devices\\\"][0][\\\"device_id\\\"]"]
    generator_cmd_file = "scripts/ascend_distributed_launcher/get_distribute_train_cmd.py"
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, generator_cmd_file))
    dataset_path = os.path.join(utils.data_root, "coco/coco2017/mindrecord_train/centernet_mindrecord")
    exec_network_shell = "cd centernet; bash scripts/run_distributed_train_ascend.sh {0} {1}"\
        .format(dataset_path, utils.rank_table_path)
    os.system(exec_network_shell)
    cmd = "ps -ef |grep train.py | grep coco | grep -v grep"
    ret = utils.process_check(120, cmd)
    assert ret
    log_file = os.path.join(cur_model_path, "LOG{}/training_log.txt")
    for i in range(8):
        per_step_time = utils.get_perf_data(log_file.format(i))
        assert per_step_time < 435
    loss_list = []
    for i in range(8):
        loss_cmd = "grep -nr \"outputs are\" {} | awk '{{print $14}}' | awk -F\")\" '{{print $1}}'"\
            .format(log_file.format(i))
        loss = utils.get_loss_data_list(log_file.format(i), cmd=loss_cmd)
        loss_list.append(loss[-1])
    assert sum(loss_list) / len(loss_list) < 58.8
