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
def test_gnmtv2_WMT_English_German():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../model_zoo/official/nlp".format(cur_path)
    model_name = "gnmt_v2"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)

    old_list = ['dataset_sink_mode=config.dataset_sink_mode']
    new_list = ['dataset_sink_mode=config.dataset_sink_mode, sink_size=100']
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "train.py"))
    old_list = ['"epochs": 6,']
    new_list = ['"epochs": 4,']
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "config/config.json"))

    mindrecord_file = "wmt16_de_en/train_tok_mindrecord/train.tok.clean.bpe.32000.en.mindrecord"
    exec_network_shell = "cd {}/scripts; sh run_distributed_train_ascend.sh {} {}"\
        .format(model_name, utils.rank_table_path, os.path.join(utils.data_root, mindrecord_file))
    ret = os.system(exec_network_shell)
    assert ret == 0

    cmd = "ps -ef | grep python | grep train.py | grep train.tok.clean.bpe | grep -v grep"
    ret = utils.process_check(120, cmd)
    assert ret

    log_file = os.path.join(cur_model_path, "scripts/device{}/log_gnmt_network{}.log")
    for i in range(8):
        per_step_time = utils.get_perf_data(log_file.format(i, i))
        print("per_step_time is", per_step_time)
        assert per_step_time < 270.0

    log_file = os.path.join(cur_model_path, "scripts/device{}/loss.log")
    loss_list = []
    for i in range(8):
        pattern1 = r"loss\: ([\d\.\+]+)\,"
        loss = utils.parse_log_file(pattern1, log_file.format(i))
        print("loss is", loss)
        loss_list.append(loss[-1])
    print("loss_list is", loss_list)
    print(sum(loss_list) / len(loss_list))
    assert sum(loss_list) / len(loss_list) < 120
