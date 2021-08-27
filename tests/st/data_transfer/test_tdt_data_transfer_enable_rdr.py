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
import tempfile
import json
import pytest

import mindspore.context as context
from .test_tdt_data_transfer import test_tdt_consume_beyond_produce

# create config file for RDR
def create_config_file(path):
    data_dict = {'rdr': {'enable': True, 'path': path}}
    filename = os.path.join(path, "mindspore_config.json")
    with open(filename, "w") as f:
        json.dump(data_dict, f)
    return filename

def test_train(device_type):
    context.set_context(mode=context.GRAPH_MODE, device_target=device_type)
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = create_config_file(tmpdir)
        context.set_context(env_config_path=config_file)
        test_tdt_consume_beyond_produce()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_train_with_Ascend():
    test_train("Ascend")

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_with_GPU():
    test_train("GPU")

@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_train_with_CPU():
    test_train("CPU")
