# Copyright 2020 Huawei Technologies Co., Ltd
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


def test_ps_ascend_multi_worker_multi_server():
    return_code = os.system("bash shell_run_test.sh Ascend 8 8 127.0.0.1 8088")
    assert return_code == 0


def test_ps_ascend():
    return_code = os.system("bash shell_run_test.sh Ascend 1 1 127.0.0.1 8088")
    assert return_code == 0


def test_ps_gpu_multi_worker_multi_server():
    return_code = os.system("bash shell_run_test.sh GPU 8 8 127.0.0.1 8088")
    assert return_code == 0


def test_ps_gpu():
    return_code = os.system("bash shell_run_test.sh GPU 1 1 127.0.0.1 8088")
    assert return_code == 0
