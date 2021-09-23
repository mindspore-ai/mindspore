# Copyright 2020 Huawei Technologies Co., Ltd
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
"""maskrcnn testing script."""

import os
import pytest
from tests.st.model_zoo_tests import utils


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_maskrcnn_export():
    """
    export maskrcnn air.
    """
    old_list = ["(config=config)", "(net, param_dict_new)"]
    new_list = ["(config=config)\\n    '''", "(net, param_dict_new)\\n    '''"]

    cur_path = os.getcwd()
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "maskrcnn"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)

    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "export.py"))
    # ckpt_path = os.path.join(utils.ckpt_root, "bgcf/bgcf_trained.ckpt")
    exec_export_shell = "cd {}; python export.py --config_path default_config.yaml".format(model_name)
    os.system(exec_export_shell)
    assert os.path.exists(os.path.join(cur_model_path, "{}.air".format(model_name)))

if __name__ == '__main__':
    test_maskrcnn_export()
