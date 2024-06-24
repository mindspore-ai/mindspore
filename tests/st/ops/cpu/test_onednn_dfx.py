# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import os
import re
import platform
from collections import defaultdict
import pytest
import mindspore.context as context


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_onednn_dfx_log():
    """
    Feature: onednn dfx
    Description: Use 'Python -O -m pytest -s xxx.py' to enable __debug__
    Expectation: begin_cnt == end_cnt
    """
    if platform.system().lower() != 'linux':
        return

    os.environ['GLOG_v'] = '0'
    log_name = './onednn_dfx.log'
    # need cd to current dir
    cmd = "pytest -s test_maxpool_op.py::test_max_pool3d_1 --disable-warnings > {} 2>&1".format(log_name)
    out = os.popen(cmd)
    out.read()
    keyword_begin = "begin to invoke"
    keyword_end = "end to invoke"
    begin_cnt = 0
    end_cnt = 0
    log_pattern = re.compile(r'\[[A-Z]+\] [A-Z]+\([\d]+\,[a-f\d]+,')
    # {tid0: [log_line0, log_line1], tid1: [log_line2, log_line3, log_line4], ...}
    multi_dict = defaultdict(list)

    with open(log_name, "r") as f:
        for line in f.readlines():
            if not log_pattern.match(line):
                continue
            # log example: '[DEBUG] KERNEL(48810, 7f42b77fd700,python):2022-01-10-16:19:13.521.086 xxxx.'
            tid = line.split(' ')[1].split(',')[1]
            line = line.replace('\n', '').replace('\r', '')
            multi_dict[tid].append(line)

    assert multi_dict.keys()  # check empty
    for tid in multi_dict.keys():
        if not __debug__:
            f = open('./onednn_dfx-{}.log'.format(tid), "w")
        for line in multi_dict[tid]:
            if not __debug__:
                f.write(line + '\n')

            if keyword_begin in line:
                begin_cnt += 1
                last_begin_line = line
            elif keyword_end in line:
                end_cnt += 1

        if begin_cnt != end_cnt:
            print("\n=======The line below has no pairing end line======")
            print(last_begin_line)
            print("===================================================")
            assert begin_cnt == end_cnt

        if not __debug__:
            f.close()
