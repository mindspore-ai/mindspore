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
import re
import subprocess

import pytest

from tests.st.model_zoo_tests import utils


def init_files():
    cur_path = os.getcwd()
    model_path = "{}/../../../../tests/models/official/audio".format(cur_path)
    model_name = "tacotron2"
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    old_list = ["from src.rnns import LSTM", r"\ h=", r"mask,\ outputs,\ self.fullzeros"]
    new_list = ["from mindspore.nn import LSTM", " ", r"mask,\ outputs,\ P.ZerosLike\(\)\(outputs\)"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/tacotron2.py"))
    # To run faster on CI, reduce some configuration parameters:
    old_list = [r"num_mels\ =\ 80"]
    new_list = [r"num_mels\ =\ 16"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/hparams.py"))
    old_list = [r"epoch_num:\ 2000"]
    new_list = [r"epoch_num:\ 2"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "ljspeech_config.yaml"))
    # The ci environment does not have some python package
    old_list = [r"from\ src.text\ import\ cleaners"]
    new_list = [r"\#from\ src.text\ import\ cleaners"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/text/__init__.py"))
    os.system("cp -r {} {}".format(os.path.join(cur_path, "run_tacotron2_dynamic.py"), cur_model_path))
    return cur_model_path


def get_res(ret_output):
    loss_list = []
    overflow_list = []
    loss_pattren = re.compile(r"loss is ([0-9\.a-zA-Z]+),")
    for line in ret_output:
        match_res = loss_pattren.search(line)
        if match_res is not None:
            loss = match_res.groups()[0]
            loss_list.append(float(loss))
            overflow_list.append("Overflow" in line)
    return loss_list, overflow_list


def run_tacotron2_dynamic_case(cur_model_path):
    run_file = os.path.join(cur_model_path, "run_tacotron2_dynamic.py")
    exec_net_cmd = f"python {run_file}"
    ret = subprocess.run(exec_net_cmd.split(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret.returncode:
        raise RuntimeError(f"fail to execute the cmd:\n{ret.stdout}")
    loss_list, overflow_list = get_res(ret.stdout.split("\n"))
    print(f"loss_list is {loss_list}, overflow_list is {overflow_list}", flush=True)
    has_overflow = any(overflow_list)
    return loss_list, has_overflow


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tacotron2_dynamic_pynative():
    """
    Feature: tacotron2_dynamic
    Description: test tacotron2_dynamic run
    Expectation: loss is same with the expect (wait matmul dynamic bug fix.)
    """
    cur_model_path = init_files()
    loss_list, has_overflow = run_tacotron2_dynamic_case(cur_model_path)
    assert not has_overflow
    assert len(loss_list) >= 2
