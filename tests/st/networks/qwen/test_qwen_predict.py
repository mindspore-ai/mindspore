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
from tests.mark_utils import arg_mark
from tests.st.networks.utils import get_num_from_log

os.environ["GLOG_v"] = "1"
os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
os.environ["MS_ENABLE_LCCL"] = "1"
os.environ["MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST"] = "MatMulAllReduce"
os.environ["CUSTOM_MATMUL_SHUFFLE"] = "on"
os.environ["RUN_MODE"] = "predict"
TOELERANCE = 5e-2
PEAK_MEMORY_NAME = "Actual peak memory usage (with fragments):"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_qwen_1p_bs1():
    """
    Feature: kbk predict
    Description: test_qwen_1p_bs1
    Expectation: AssertionError
    """
    test_case = "test_qwen_1p_bs1"
    device_id = os.getenv('DEVICE_ID', '0')
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch.sh {sh_path}/configs/predict_qwen1.5.yaml 1 {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"cat {log_path}")
    assert ret == 0

    expect_peak_memory = 3976
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_qwen_1p_bs4():
    """
    Feature: kbk predict
    Description: test_qwen_1p_bs4
    Expectation: AssertionError
    """
    test_case = "test_qwen_1p_bs4"
    device_id = os.getenv('DEVICE_ID', '0')
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch.sh {sh_path}/configs/predict_qwen1.5.yaml 1 {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"cat {log_path}")
    assert ret == 0

    expect_peak_memory = 3978
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_qwen_4p_bs1():
    """
    Feature: kbk predict
    Description: test_qwen_4p_bs1
    Expectation: AssertionError
    """
    test_case = "test_qwen_4p_bs1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch.sh {sh_path}/configs/predict_qwen1.5.yaml 4 {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"cat {log_path}")
    assert ret == 0

    expect_peak_memory = 3385
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_qwen_4p_bs4():
    """
    Feature: Trainer.predict()
    Description: Test trainer for predict.
    Expectation: AssertionError
    """
    test_case = "test_qwen_4p_bs4"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch.sh {sh_path}/configs/predict_qwen1.5.yaml 4 {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"cat {log_path}")
    assert ret == 0

    expect_peak_memory = 3385
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_qwen_4p_bs4_bf16():
    """
    Feature: kbk predict
    Description: test_qwen_4p_bs4_bf16
    Expectation: AssertionError
    """
    test_case = "test_qwen_4p_bs4_bf16"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch.sh {sh_path}/configs/predict_qwen1.5.yaml 4 {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"cat {log_path}")
    assert ret == 0

    expect_peak_memory = 3385
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)
