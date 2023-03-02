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
def test_DeeplabV3_voc2007():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "deeplabv3"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)

    old_list = ['/PATH_TO_DATA/vocaug/vocaug_mindrecord/vocaug_mindrecord0',
                '/PATH/TO/PRETRAIN_MODEL']
    new_list = [os.path.join(utils.data_root, "voc/voc2012/mindrecord_train/vocaug_mindrecord0"),
                os.path.join(utils.ckpt_root, "deeplabv3/resnet101_ascend.ckpt")]
    utils.exec_sed_command(old_list, new_list,
                           os.path.join(cur_model_path, "scripts/run_distribute_train_s16_r1.sh"))

    old_list = ['model.train(args.train_epochs',
                'callbacks=cbs', ', save_graphs=False,']
    new_list = ['model.train(30',
                'callbacks=cbs, sink_size=2', ',']
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "train.py"))

    exec_network_shell = "cd {}/scripts; sh run_distribute_train_s16_r1.sh {}".format(
        model_name, utils.rank_table_path)
    ret = os.system(exec_network_shell)
    assert ret == 0
    cmd = "ps -ef --columns 1000 | grep python | grep train.py | grep -v grep"
    ret = utils.process_check(100, cmd)
    assert ret

    log_file = os.path.join(cur_model_path, "scripts/s16_aug_train/device{}/log")
    for i in range(8):
        per_step_time = utils.get_perf_data(log_file.format(i))
        print("per_step_time is", per_step_time)
        assert per_step_time < 585.0
    loss_list = []
    for i in range(8):
        loss = utils.get_loss_data_list(log_file.format(i))
        print("loss is", loss[-1])
        loss_list.append(loss[-1])
    assert sum(loss_list) / len(loss_list) < 3.5
