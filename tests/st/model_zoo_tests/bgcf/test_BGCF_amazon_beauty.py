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


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_BGCF_amazon_beauty():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../model_zoo/official/gnn".format(cur_path)
    model_name = "bgcf"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)

    old_list = ["--datapath=../data_mr"]
    new_list = ["--datapath={}".format(os.path.join(utils.data_root, "amazon_beauty/mindrecord_train"))]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "scripts/run_train_ascend.sh"))
    old_list = ["default=600,"]
    new_list = ["default=50,"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/config.py"))
    old_list = ["context.set_context(device_id=int(parser.device))"]
    new_list = ["context.set_context()"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "train.py"))
    exec_network_shell = "cd {}/scripts; bash run_train_ascend.sh".format(model_name)
    ret = os.system(exec_network_shell)
    assert ret == 0

    cmd = "ps -ef|grep python |grep train.py|grep amazon_beauty|grep -v grep"
    ret = utils.process_check(300, cmd)
    assert ret

    log_file = os.path.join(cur_model_path, "scripts/train/log")
    pattern1 = r"loss ([\d\.\+]+)\,"
    loss_list = utils.parse_log_file(pattern1, log_file)
    loss_list = loss_list[-5:]
    print("last 5 epoch average loss is", sum(loss_list) / len(loss_list))
    assert sum(loss_list) / len(loss_list) < 6400

    pattern1 = r"cost:([\d\.\+]+)"
    epoch_time_list = utils.parse_log_file(pattern1, log_file)[1:]
    print("per epoch time:", sum(epoch_time_list) / len(epoch_time_list))
    assert sum(epoch_time_list) / len(epoch_time_list) < 2.2


def test_bgcf_export_mindir():
    cur_path = os.getcwd()
    model_path = "{}/../../../../model_zoo/official/gnn".format(cur_path)
    model_name = "bgcf"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)

    ckpt_path = os.path.join(utils.ckpt_root, "bgcf/bgcf_trained.ckpt")
    exec_export_shell = "cd {}; python export.py --ckpt_file={} --file_format=MINDIR".format(model_name, ckpt_path)
    os.system(exec_export_shell)
    assert os.path.exists(os.path.join(cur_model_path, "{}.mindir".format(model_name)))
