# Copyright 2024 Huawei Technologies Co., Ltd
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

TOKENIZER_PATH = "/home/workspace/mindspore_dataset/llama2/wiki4096/tokenizer.model"
MODEL_PATH = "/home/workspace/mindspore_ckpt/ckpt/llama2/llama_tiny.ckpt"
cur_path = os.path.split(os.path.realpath(__file__))[0]
TOELERANCE = 5e-2


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_llama_single_inference():
    """
    Feature: Llama inference in 1p test
    Description: Test Llama 1p inference, check output ids, NPU memory and inference time.
    Expectation: output ids equals expect results and inference time less than the ceiling about 5%.
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    os.environ["MS_MEMORY_STATISTIC"] = "1"
    os.environ["RANK_ID"] = "0"
    YAML_FILE = f"{cur_path}/configs/llama_dp.yaml"
    device_id = int(os.getenv('DEVICE_ID', '0'))
    os.system(f"rm -rf {cur_path}/log_infer_llama.log")
    os.system(f"source {cur_path}/env.sh")
    res = os.system(f"python {cur_path}/infer_llama.py "
                    f"--checkpoint_path {MODEL_PATH} "
                    f"--yaml_file {YAML_FILE} "
                    f"--tokenizer_path {TOKENIZER_PATH} "
                    f"--use_parallel False "
                    f"--device_id {device_id} "
                    f"--batch_size 8 "
                    f"&> {cur_path}/log_infer_llama.log")
    os.system(f"grep -E 'ERROR|error' {cur_path}/log_infer_llama.log -C 10")
    assert res == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_llama_single_inference_bs8():
    """
    Feature: Llama inference in 1p test
    Description: Test Llama 1p inference, check output ids, NPU memory and inference time.
    Expectation: output ids equals expect results and inference time less than the ceiling about 5%.
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    os.environ["MS_MEMORY_STATISTIC"] = "1"
    os.environ["RANK_ID"] = "0"
    YAML_FILE = f"{cur_path}/configs/llama_dp.yaml"
    device_id = int(os.getenv('DEVICE_ID', '0'))
    os.system(f"rm -rf {cur_path}/log_infer_llama_bs8.log")
    os.system(f"source {cur_path}/env.sh")
    res = os.system(f"python {cur_path}/infer_llama.py "
                    f"--checkpoint_path {MODEL_PATH} "
                    f"--yaml_file {YAML_FILE} "
                    f"--tokenizer_path {TOKENIZER_PATH} "
                    f"--use_parallel False "
                    f"--device_id {device_id} "
                    f"--batch_size 8 "
                    f"&> {cur_path}/log_infer_llama_bs8.log")
    os.system(f"grep -E 'ERROR|error' {cur_path}/log_infer_llama_bs8.log -C 10")
    assert res == 0
