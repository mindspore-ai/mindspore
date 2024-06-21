# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore as ms
from tests.st.networks import utils
from tests.st.utils import test_utils

ms.set_seed(1)
np.random.seed(1)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@test_utils.run_test_with_On
def test_resnet50_boost_imagenet2012_ascend():
    """
    Feature: Resnet50 boost in ge process
    Description: test_ge_resnet50_imagenet2012_ascend
    Expectation: Success
    """
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/cv".format(current_path)
    model = "resnet"
    utils.copy_files(model_path, current_path, model)
    cur_model_path = os.path.join(current_path, "resnet")
    list_old = ["config.epoch_size - config.pretrain_epoch_size", "=dataset.get_dataset_size()", "=dataset_sink_mode",
                "\\\"total_steps\\\""]
    list_new = ["1", "=1", "=True", r"\\\"param_groups\\\"\: 2, \\\"total_steps\\\""]
    utils.exec_sed_command(list_old, list_new, os.path.join(cur_model_path, "train.py"))
    list_old = ["ms.set_seed(1)"]
    list_new = ['ms.set_seed(1)\\nms.set_context(jit_level=\\"O2\\")']
    utils.exec_sed_command(list_old, list_new, os.path.join(cur_model_path, "train.py"))
    old_list = ["from mindspore._checkparam import Validator"]
    new_list = ["from mindspore import _checkparam as Validator"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/momentum.py"))
    dataset = os.path.join(utils.data_root, "imagenet/imagenet_original/train")
    # Do not execute ckpt graph
    config = os.path.join(cur_model_path, "config", "resnet50_imagenet2012_Boost_config.yaml")
    list_old = ["save_checkpoint: True"]
    list_new = ["save_checkpoint: False"]
    utils.exec_sed_command(list_old, list_new, config)
    exec_network_shell = "cd {}/resnet/scripts; bash run_standalone_train.sh {} {}" \
        .format(current_path, dataset, config)
    os.system(exec_network_shell)
    cmd = "ps -ef | grep python | grep train.py | grep -v grep"
    result = utils.process_check(120, cmd)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'
    assert result
    log_file = os.path.join(cur_model_path, "scripts/train/log")
    loss_list = utils.get_loss_data_list(log_file)
    assert round(loss_list[-1]) <= 7
