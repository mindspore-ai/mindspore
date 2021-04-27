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
def test_ncf():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../model_zoo/official/recommend".format(cur_path)
    model_name = "ncf"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    old_list = ["train_epochs 20"]
    new_list = ["train_epochs 4"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "scripts/run_train.sh"))
    old_list = ["with open(cache_path, \\\"wb\\\")", "pickle.dump"]
    new_list = ["\\# with open(cache_path, \\\"wb\\\")", "\\# pickle.dump"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/dataset.py"))
    dataset_path = os.path.join(utils.data_root, "MovieLens")
    exec_network_shell = "cd ncf; bash scripts/run_train.sh {0} checkpoint/ > train.log 2>&1 &"\
        .format(dataset_path)
    os.system(exec_network_shell)
    cmd = "ps -ef|grep python|grep train.py|grep train_epochs|grep -v grep"
    ret = utils.process_check(100, cmd)
    assert ret
    log_file = os.path.join(cur_model_path, "train.log")
    per_step_time = utils.get_perf_data(log_file)
    assert per_step_time < 2.0
    loss = utils.get_loss_data_list(log_file)[-1]
    assert loss < 0.33
