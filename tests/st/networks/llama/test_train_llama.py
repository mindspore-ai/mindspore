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

import numpy as np
import pytest

from tests.st.networks.utils import get_num_from_log

DATASET_PATH = "/home/workspace/mindspore_dataset/llama2/wiki512"
MODEL_PATH = "/home/workspace/mindspore_ckpt/ckpt/llama2/llama_tiny.ckpt"
cur_path = os.path.split(os.path.realpath(__file__))[0]
TOELERANCE = 5e-2
TOKEN_THRES = 3

def check_result(sh_path, exp_res):
    """
    check method to get result
    """
    static_memory = get_num_from_log(f"{sh_path}/llama_finetune*/finetune_llama_log*log", "Total Static Memory size:")
    dyn_memory = get_num_from_log(f"{sh_path}/llama_finetune*/finetune_llama_log*log", "Total Dynamic memory size:")
    peak_memory = get_num_from_log(f"{sh_path}/llama_finetune*/finetune_llama_log*log", "Actual peak memory usage:")

    actual_throughout = get_num_from_log(f"{sh_path}/llama_finetune*/finetune_llama_log*log", "Actual Throughout is:")
    loss = get_num_from_log(f"{sh_path}/llama_finetune*/finetune_llama_log*log", "Final loss is:", is_loss=True)

    EXP_STATIC_MEM, EXP_DYN_MEM, EXP_PEAK_MEM, EXP_THROUGHOUT, EXP_LOSS = exp_res

    assert static_memory <= EXP_STATIC_MEM * (1 + TOELERANCE)
    assert dyn_memory <= EXP_DYN_MEM * (1 + TOELERANCE)
    assert peak_memory <= EXP_PEAK_MEM * (1 + TOELERANCE)
    assert actual_throughout >= EXP_THROUGHOUT - TOKEN_THRES

    np.testing.assert_allclose(loss, EXP_LOSS, rtol=5e-3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.platform_x86_ascend910b_training
@pytest.mark.env_single
def test_llama_data_parallel():
    """
    Feature: Llama pure data parallel in 8p test
    Description: Test Llama 8p training, check loss in 20 steps, NPU memory, throughout.
    Expectation: loss stable in 0.5% range, memory smaller than ceiling about 5%, throughout larger than floor about 5%
    """
    ret = os.system(f"sh {cur_path}/run_semi_parallel_llama_train.sh llama_dp.yaml {DATASET_PATH} {MODEL_PATH}")
    assert ret == 0
    exp_loss = [10.447928, 10.429884, 10.242577, 9.8807745, 9.467778, 9.319356, 8.988123, 8.55187, 8.557504, 8.247183]
    exp_res = (1687, 1373, 3061, 36, exp_loss)
    check_result(cur_path, exp_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.platform_x86_ascend910b_training
@pytest.mark.env_single
def test_llama_data_parallel_optim_cut_steps():
    """
    Feature: Llama pure data parallel with half optimizer cut in 8p test
    Description: Test Llama 8p training, check loss in 20 steps, NPU memory, throughout.
    Expectation: loss stable in 0.5% range, memory smaller than ceiling about 5%, throughout larger than floor about 5%
    """
    ret = os.system(f"sh {cur_path}/run_semi_parallel_llama_train.sh llama_dp_optim_cut.yaml "
                    f"{DATASET_PATH} {MODEL_PATH}")
    assert ret == 0
    exp_loss = [10.447928, 10.429836, 10.242679, 9.881067, 9.468317, 9.320203, 8.989393, 8.553177, 8.558171, 8.248666]
    exp_res = (1881, 1744, 3625, 19.4, exp_loss)
    check_result(cur_path, exp_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.platform_x86_ascend910b_training
@pytest.mark.env_single
def test_llama_pipeline_parallel():
    """
    Feature: Llama pipeline parallel in lazyinline mode in 8p test
    Description: Test Llama 8p training, check loss in 20 steps, NPU memory, throughout.
    Expectation: loss stable in 0.5% range, memory smaller than ceiling about 5%, throughout larger than floor about 5%,
    compile time less than 200% of itself.
    """
    ret = os.system(f"sh {cur_path}/run_semi_parallel_llama_train.sh llama_pp.yaml {DATASET_PATH} {MODEL_PATH}")
    assert ret == 0
    compile_time = get_num_from_log(f"{cur_path}/llama_finetune*/finetune_llama_log*log", "Compile time:")
    assert compile_time < 95
    exp_loss = [10.448, 10.430, 10.242, 9.881, 9.467, 9.319, 8.987, 8.550, 8.557, 8.245]
    exp_res = (1748, 463, 2129, 27.5, exp_loss)
    check_result(cur_path, exp_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.platform_x86_ascend910b_training
@pytest.mark.env_single
def test_llama_gradient_accumulate_steps():
    """
    Feature: Llama data parallel, model parallel with accumulating gradient step in 8p test
    Description: Test Llama 8p training, check loss in 20 steps, NPU memory, throughout.
    Expectation: loss stable in 0.5% range, memory smaller than ceiling about 5%, throughout larger than floor about 5%
    """
    ret = os.system(f"sh {cur_path}/run_semi_parallel_llama_train.sh llama_accum.yaml {DATASET_PATH} {MODEL_PATH}")
    assert ret == 0
    exp_loss = [10.447860, 10.440143, 10.265631, 9.911415, 9.493698, 9.331136, 8.995336, 8.557518, 8.565901, 8.252847]
    exp_res = (2085, 1169, 3254, 24.8, exp_loss)
    check_result(cur_path, exp_res)
