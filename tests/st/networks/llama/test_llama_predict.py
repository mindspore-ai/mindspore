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
from tests.st.networks.utils import get_num_from_log
from tests.mark_utils import arg_mark

os.environ["GLOG_v"] = "1"
os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
os.environ["MS_ENABLE_LCCL"] = "1"
os.environ["MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST"] = "MatMulAllReduce"
os.environ["CUSTOM_MATMUL_SHUFFLE"] = "on"
os.environ["RUN_MODE"] = "predict"
TOELERANCE = 5e-2
PEAK_MEMORY_NAME = "Actual peak memory usage (with fragments):"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_llama_predict_1p_bs1():
    """
    Feature: kbk predict
    Description: test_llama_predict_1p_bs1
    Expectation: AssertionError
    """
    test_case = "test_llama_1p_bs1"
    device_id = os.getenv('DEVICE_ID', '0')
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch_llama.sh {sh_path}/configs/predict_llama_70b.yaml 1 predict {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"grep -E 'ERROR|error' {log_path} -C 10")
    assert ret == 0

    expect_peak_memory = 5079
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_llama_predict_1p_bs4():
    """
    Feature: kbk predict
    Description: test_llama_predict_1p_bs4
    Expectation: AssertionError
    """
    test_case = "test_llama_1p_bs4"
    device_id = os.getenv('DEVICE_ID', '0')
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch_llama.sh {sh_path}/configs/predict_llama_70b.yaml 1 predict {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"grep -E 'ERROR|error' {log_path} -C 10")
    os.system(f"cat {log_path}")
    assert ret == 0

    expect_peak_memory = 5083
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_llama_predict_4p_bs1():
    """
    Feature: kbk predict
    Description: test_llama_predict_4p_bs1
    Expectation: AssertionError
    """
    test_case = "test_llama_4p_bs1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch_llama.sh {sh_path}/configs/predict_llama_70b.yaml 4 predict {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"grep -E 'ERROR|error' {log_path} -C 10")
    assert ret == 0

    expect_peak_memory = 2630
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_llama_predict_4p_bs4():
    """
    Feature: kbk predict
    Description: test_llama_predict_4p_bs4
    Expectation: AssertionError
    """
    test_case = "test_llama_4p_bs4"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch_llama.sh {sh_path}/configs/predict_llama_70b.yaml 4 predict {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"grep -E 'ERROR|error' {log_path} -C 10")
    assert ret == 0

    expect_peak_memory = 2630
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_llama_predict_4p_bs4_bf16():
    """
    Feature: kbk predict
    Description: test_llama_predict_4p_bs4_bf16
    Expectation: AssertionError
    """
    test_case = "test_llama_4p_bs4_bf16"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch_llama.sh {sh_path}/configs/predict_llama_70b.yaml 4 predict {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"grep -E 'ERROR|error' {log_path} -C 10")
    assert ret == 0

    expect_peak_memory = 2630
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_llama_predict_4p_bs4_w8a16():
    """
    Feature: kbk predict
    Description: test_llama_predict_4p_bs4_w8a16
    Expectation: AssertionError
    """
    test_case = "test_llama_4p_bs4_w8a16"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(
        f"bash {sh_path}/mpirun_launch_llama.sh {sh_path}/configs/predict_llama_70b.yaml 4 predict {test_case}")
    log_path = f"{sh_path}/{test_case}.log"
    os.system(f"grep -E 'ERROR|error' {log_path} -C 10")
    assert ret == 0

    expect_peak_memory = 2630
    peak_memory = get_num_from_log(f"{log_path}", PEAK_MEMORY_NAME)
    assert peak_memory <= expect_peak_memory * (1 + TOELERANCE)
