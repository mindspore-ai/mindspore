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
import subprocess
import mindspore as ms


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_msrun():
    """
    Feature: 'msrun' launch utility.
    Description: Launch distributed training job with dynamic cluster using msrun.
    Expectation: All workers are successfully spawned and running training.
    """
    ms.set_context(jit_level='O0')
    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "\
        "--master_port=10969 --join=True "\
        "test_msrun.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    assert return_code == 0


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_msrun_exception():
    """
    Feature: 'msrun' launch utility.
    Description: Create python and cpp exception for msrun respectively and check whether cluster could exit
                 and filter out the error logs.
    Expectation: Cluster exits with no process hanging and error log is filtered out.
    """
    # Need to set log level so key words could be filtered out.
    os.environ['GLOG_v'] = str(2)
    result = subprocess.getoutput(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "\
        "--master_port=10969 --join=True --log_dir=python_exception_log "\
        "test_msrun_exception.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist "\
        "--exception_type='python'"
    )
    assert result.find("Rank 0 throw python exception.") != -1
    assert result.find("The node: 0 is timed out") != -1


    result = subprocess.getoutput(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "\
        "--master_port=10969 --join=True --log_dir=cpp_exception_log "\
        "test_msrun_exception.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist "\
        "--exception_type='cpp'"
    )
    assert result.find("For 'MatMul' the input dimensions must be equal, but got 'x1_col': 84 and 'x2_row': 64") != -1
    assert result.find("The node: 1 is timed out") != -1
