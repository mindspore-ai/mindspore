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
"""
Test senv_recv interface
"""
import os

from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_tre_step_resume_training():
    """
    Feature: Test TREError step resuming training.
    Description: Test TREError step resuming training..
    Expectation: The loss after resume training is same as error occurred before.
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"bash {sh_path}/tre_step_resume_run.sh")
    assert ret == 0
