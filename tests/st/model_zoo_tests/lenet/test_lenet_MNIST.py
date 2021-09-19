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


@pytest.mark.level2
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_lenet_MNIST():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "lenet"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    train_log = os.path.join(cur_model_path, "train_ascend.log")
    ckpt_file = os.path.join(cur_model_path, "ckpt/checkpoint_lenet-10_1875.ckpt")
    infer_log = os.path.join(cur_model_path, "infer_ascend.log")
    dataset_path = os.path.join(utils.data_root, "mnist")
    exec_network_shell = "cd {0}; python train.py --data_path={1} > {2} 2>&1"\
        .format(model_name, dataset_path, train_log)
    ret = os.system(exec_network_shell)
    assert ret == 0
    exec_network_shell = "cd {0}; python eval.py --data_path={1} --ckpt_path={2} > {3} 2>&1"\
        .format(model_name, dataset_path, ckpt_file, infer_log)
    ret = os.system(exec_network_shell)
    assert ret == 0

    per_step_time = utils.get_perf_data(train_log)
    print("per_step_time is", per_step_time)
    assert per_step_time < 1.5

    pattern = r"'Accuracy': ([\d\.]+)}"
    acc = utils.parse_log_file(pattern, infer_log)
    print("acc is", acc)
    assert acc[0] > 0.98
