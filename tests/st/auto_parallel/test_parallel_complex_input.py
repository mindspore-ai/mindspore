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

os.environ['HCCL_IF_BASE_PORT'] = '30000'

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards")
def test_graph_mode_parallel_complex_input():
    '''
    Feature: Parallel Support for Complex64 input
    Description: graph mode
    Expectation: Run success
    '''
    os.environ['ASCEND_GLOBAL_EVENT_ENABLE'] = '1'
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '1'
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    os.environ['GLOG_v'] = '1'
    os.environ['MS_SUBMODULE_LOG_v'] = r'{RUNTIME_FRAMEWORK:0}'

    # print("netstat")
    # os.system(f"netstat")
    print("netstat -tunlp")
    os.system("netstat -tunlp")
    os.system("netstat -tunlp > netstat.txt")

    os.system("rm -rf ~/ascend")
    cmd = ("msrun "
           "--worker_num=8 "
           "--local_worker_num=8 "
           "--master_port=10001 "
           "--join=True "
           "--log_dir=log_output "
           "python parallel_complex_input.py")
    print(f"cmd is:\n{cmd}")
    ret = os.system(cmd)

    if ret != 0:
        import datetime
        t = datetime.datetime.now()
        f = t.strftime('%m%d%H%M%S')
        os.system(f"mkdir ~/parallel_complex_input_{f}")
        os.system(f"cp -rf log_output ~/parallel_complex_input_{f}")
        os.system(f"cp -rf ~/ascend/log ~/parallel_complex_input_{f}")
        os.system(f"cp -rf netstat.txt ~/parallel_complex_input_{f}")
    assert ret == 0
