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
"""
Test module for testing the paralleled llama interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_llama_model/test_parallel_predict.py
"""
import os

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_parallel_predict():
    """
    Feature: Trainer.predict()
    Description: Test trainer for predict.
    Expectation: AssertionError
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    os.environ["MS_ENABLE_LCCL"] = "1"
    os.environ["MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST"] = "MatMulAllReduce"
    os.environ["CUSTOM_MATMUL_SHUFFLE"] = "on"
    os.environ["RUN_MODE"] = "predict"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"source {sh_path}/env.sh")
    ret = os.system(f"bash {sh_path}/mpirun_launch_llama.sh 4 test_predict")
    os.system(f"grep -E 'ERROR|error' {sh_path}/test_predict.log -C 10")
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_parallel_predict_bf16():
    """
    Feature: Trainer.predict()
    Description: Test trainer for bfloat16 predict.
    Expectation: AssertionError
    """
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    os.environ["MS_ENABLE_LCCL"] = "1"
    os.environ["MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST"] = "MatMulAllReduce"
    os.environ["CUSTOM_MATMUL_SHUFFLE"] = "on"
    os.environ["RUN_MODE"] = "predict"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"source {sh_path}/env.sh")
    ret = os.system(
        f"bash {sh_path}/mpirun_launch_llama.sh 4 test_predict_bf16")
    os.system(f"grep -E 'ERROR|error' {sh_path}/test_predict_bf16.log -C 10")
    assert ret == 0
