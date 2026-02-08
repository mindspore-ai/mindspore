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
""" test join error """
import subprocess
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_st_ms_join_error_0003():
    """
    Feature: Error info and trace info for join error.
    Description: Check error info and trace info for join error case.
    Expectation: No exception.
    """
    case_num = 3
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "./ms_join_error_code.py"
    log_path = os.path.join(dir_path, file_name)
    execute_script = f"python {log_path} -c={case_num} " \
                     f"> abnormal_code_000{case_num}.log 2>&1"
    subprocess.run(execute_script, shell=True)

    with open(f"abnormal_code_000{case_num}.log", encoding="utf-8") as f:
        log = f.read()

    excepted_err_msg = "Type Join Failed: dtype1 = String, dtype2 = Int64"

    err_msg1 = "return '2'"
    cmd1 = f'grep -b1 "{err_msg1}" abnormal_code_000{case_num}.log'
    sub = subprocess.Popen(args=cmd1, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout_data, _ = sub.communicate()
    error_list1 = stdout_data.strip().split('\n')
    err_msg2 = 'return 1'
    cmd2 = f"grep -b1 '{err_msg2}' abnormal_code_000{case_num}.log"
    sub1 = subprocess.Popen(args=cmd2, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    stdout_data1, _ = sub1.communicate()
    error_list2 = stdout_data1.strip().split('\n')

    assert error_list1[-2].index('r') == error_list1[-1].index('^')
    assert error_list2[-2].index('r') == error_list2[-1].index('^')
    assert excepted_err_msg in log
    os.remove(f"abnormal_code_000{case_num}.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_st_ms_join_error_0004():
    """
    Feature: Error info and trace info for join error.
    Description: Check error info and trace info for join error case.
    Expectation: No exception.
    """
    case_num = 4
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "./ms_join_error_code.py"
    log_path = os.path.join(dir_path, file_name)
    execute_script = f"python {log_path} -c={case_num} " \
                     f"> abnormal_code_000{case_num}.log 2>&1"
    subprocess.run(execute_script, shell=True)
    with open(f"abnormal_code_000{case_num}.log", encoding="utf-8") as f:
        log = f.read()

    excepted_err_msg = "Type Join Failed: Abstract type AbstractRefTensor cannot join with AbstractScalar"

    err_msg1 = 'return input_x'
    cmd1 = f"grep -b1 '{err_msg1}' abnormal_code_000{case_num}.log"
    sub = subprocess.Popen(args=cmd1, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout_data, _ = sub.communicate()
    error_list1 = stdout_data.strip().split('\n')
    err_msg2 = 'return 2'
    cmd2 = f"grep -b1 '{err_msg2}' abnormal_code_000{case_num}.log"
    sub1 = subprocess.Popen(args=cmd2, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    stdout_data1, _ = sub1.communicate()
    error_list2 = stdout_data1.strip().split('\n')

    assert error_list1[-2].index('r') == error_list1[-1].index('^')
    assert error_list2[-2].index('r') == error_list2[-1].index('^')
    assert excepted_err_msg in log
    os.remove(f"abnormal_code_000{case_num}.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_st_ms_join_error_0005():
    """
    Feature: Error info and trace info for join error.
    Description: Check error info and trace info for join error case.
    Expectation: No exception.
    """
    case_num = 5
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "./ms_join_error_code.py"
    log_path = os.path.join(dir_path, file_name)
    execute_script = f"python {log_path} -c={case_num} " \
                     f"> abnormal_code_000{case_num}.log 2>&1"
    subprocess.run(execute_script, shell=True)
    with open(f"abnormal_code_000{case_num}.log", encoding="utf-8") as f:
        log = f.read()

    excepted_err_msg = "TypeError: Cannot join the return values of different branches, perhaps you need to make " \
                       "them equal.\nType Join Failed: Abstract type AbstractTensor cannot join with AbstractList."

    err_msg1 = r'return x \% 2'
    cmd1 = f"grep -b1 '{err_msg1}' abnormal_code_000{case_num}.log"
    sub = subprocess.Popen(args=cmd1, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout_data, _ = sub.communicate()
    error_list1 = stdout_data.strip().split('\n')
    err_msg2 = r'return x \* 2'
    cmd2 = f"grep -b1 '{err_msg2}' abnormal_code_000{case_num}.log"
    sub1 = subprocess.Popen(args=cmd2, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    stdout_data1, _ = sub1.communicate()
    error_list2 = stdout_data1.strip().split('\n')

    assert error_list1[-2].index('r') == error_list1[-1].index('^')
    assert error_list2[-2].index('r') == error_list2[-1].index('^')
    assert excepted_err_msg in log
    os.remove(f"abnormal_code_000{case_num}.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_st_ms_join_error_0007():
    """
    Feature: Join error for list comprehension.
    Description: Check join error info for list comprehension.
    Expectation: No exception.
    """
    case_num = 7
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "./ms_join_error_code.py"
    log_path = os.path.join(dir_path, file_name)
    execute_script = f"python {log_path} -c={case_num} " \
                     f"> join_code_000{case_num}.log 2>&1"
    subprocess.run(execute_script, shell=True)
    with open(f"join_code_000{case_num}.log", encoding="utf-8") as f:
        log = f.read()
    msg1 = "Tensor(shape=[], dtype=Int64, value=18)\n" \
           "Tensor(shape=[], dtype=Int64, value=36)\n" \
           "Tensor(shape=[], dtype=Int64, value=54)\n" \
           "Tensor(shape=[], dtype=Int64, value=72)\n" \
           "Tensor(shape=[], dtype=Int64, value=90)"
    msg2 = "<class 'list'>"
    assert msg1 and msg2 in log
    os.remove(f"join_code_000{case_num}.log")
