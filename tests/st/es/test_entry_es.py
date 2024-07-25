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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_es_ascend():
    """
    Feature: Parameter Server.
    Description: Test es for Ascend.
    Expectation: success.
    """
    return_code = os.system("bash shell_run_test.sh")
    if return_code != 0:
        os.system(f"echo '\n**************** ES Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./es.log")
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='unessential')
def test_es_external_api_ascend():
    """
    Feature: Parameter Server.
    Description: Test es external api for Ascend.
    Expectation: success.
    """
    return_code = os.system("bash run_msrun.sh")
    if return_code != 0:
        os.system(f"echo '\n**************** ES External API Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./worker_0.log")
    assert return_code == 0
