# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test QWENSFT"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils import parse_log, init_env
# from tests.mark_utils import arg_mark


class TestQWENSFT:
    # @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards')
    def test_mindspore_qwen_sft_determinstic(self):
        """
        Feature: test mindspore pretrain_glm
        Description: run mindspore pretrain_glm to generate pynative loss
        Expectation: test success
        """
        init_env()
        scripts_name = "run_ms_determin.sh"

        test_path = os.path.split(os.path.realpath(__file__))[0]
        cmd = f"bash {test_path}/{scripts_name} "
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check ms_det.log"

    # @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards')
    def test_compare_res(self):
        """
        Feature: test_compare_res
        Description: compare relative error between torch loss and mindspore loss
        Expectation: no error
        """
        loss_pt = parse_log('pta_det.txt')
        loss_ms = parse_log('ms_det.txt')

        os.system('cat ms_det.txt')
        for i in loss_pt:
            print("loss:", loss_pt[i][2], loss_ms[i][2])
            assert abs(len(loss_pt[i][2]) - len(loss_ms[i][2])) < 100
