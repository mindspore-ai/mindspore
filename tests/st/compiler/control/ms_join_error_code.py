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
""" join error code """
import argparse
from mindspore import jit
from mindspore import Tensor


@jit
def ms_join_error_0003(input_x):
    if input_x < 10:
        if input_x > 0:
            if input_x > 1:  # pylint: disable=no-else-return
                return '2'
            else:
                return 1
        return [1]
    return (1,)


@jit
def ms_join_error_0004(input_x):
    for _ in range(3):
        while input_x < 10:
            input_x += 2
            if input_x >= 0:  # pylint: disable=no-else-return
                return input_x
            else:
                return 2
        return 2
    return 2


@jit
def ms_join_error_0005(input_x):
    for _ in range(3):
        x = 1
        while x < 10:
            input_x += 2
            if input_x < 5:  # pylint: disable=no-else-return
                x = x * input_x
                return x % 2
            else:
                x = [x + input_x]
                return x * 2
        return x * 2
    return  input_x


@jit
def ms_join_error_0007(input_x):
    res = [x * input_x for x in range(1, 11) if x % 2 == 0]
    print(res)
    print(type(res))
    return res


if __name__ == '__main__':
    cases_dict = {3:
                  ms_join_error_0003,
                  4:
                      ms_join_error_0004,
                  5:
                      ms_join_error_0005,
                  7:
                      ms_join_error_0007}

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case_num', type=int,
                        help='Please select your case name')
    args = parser.parse_args()
    case_num = args.case_num
    cases_dict.get(case_num)(Tensor(9))
