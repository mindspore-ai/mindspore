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
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_retinaface_resnet50():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "retinaface_resnet50"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    train_data_path = os.path.join(utils.data_root, "widerface/label.txt")
    pretrain_ckpt_path = os.path.join(utils.ckpt_root, "resnet/resnet-90_4p.ckpt")
    weight_ckpt_path = os.path.join(utils.ckpt_root, "retinaface_resnet/retinaface_res50_epoch_0.ckpt")
    if not os.path.exists(weight_ckpt_path):
        raise RuntimeError("file {} not exist.".format(weight_ckpt_path))
    if not os.path.exists(pretrain_ckpt_path):
        raise RuntimeError("file {} not exist.".format(pretrain_ckpt_path))
    old_list = ["'epoch': 100,",
                "'save_checkpoint_steps': 2000,",
                "'training_dataset': './data/widerface/train/label.txt',",
                "'pretrain_path': './data/res50_pretrain.ckpt'",
                "'resume_net': None,"]
    new_list = ["'epoch': 1,",
                "'save_checkpoint_steps': 402,",
                "'training_dataset': '{}',".format(train_data_path),
                "'pretrain_path': '{}'".format(pretrain_ckpt_path),
                "'resume_net': '{}',".format(weight_ckpt_path)]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/config.py"))
    old_list = ["sink_mode=True", "model.train(max_epoch,"]
    new_list = ["sink_mode=True, sink_size=100", "model.train(4,"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "train.py"))
    old_list = ["python train.py"]
    new_list = ["python train.py --distributed 1 --device_target GPU"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "scripts/run_distribute_gpu_train.sh"))
    exec_network_shell = "cd retinaface_resnet50; bash scripts/run_distribute_gpu_train.sh 4 1,2,3,4"
    os.system(exec_network_shell)
    cmd = "ps -ef | grep train.py | grep distributed | grep device_target | grep -v grep"
    ret = utils.process_check(120, cmd)
    if not ret:
        cmd = "{} | awk -F' ' '{{print $2}}' | xargs kill -9".format(cmd)
        os.system(cmd)
    assert ret
    log_file = os.path.join(cur_model_path, "train.log")
    pattern = r"per step time: ([\d\.]+) ms"
    per_step_time_list = utils.parse_log_file(pattern, log_file)[4:]
    print("per_step_time_list is", per_step_time_list)
    assert sum(per_step_time_list)/len(per_step_time_list) < 673.7
    loss_list = utils.get_loss_data_list(log_file)[-4:]
    print("loss_list is", loss_list)
    assert sum(loss_list) / len(loss_list) < 12.57
