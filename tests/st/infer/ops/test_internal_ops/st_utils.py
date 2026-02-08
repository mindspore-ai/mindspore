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
"""
Utility functions for test_internal_ops.
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor


def custom_compare(output, expect, mstype):
    if mstype == ms.float16:
        limit = 0.004
    elif mstype == ms.bfloat16:
        limit = 0.03
    elif mstype == ms.int8:
        limit = 0.01
    else:
        limit = 0.01  # default limit

    #print("limit = ", limit)
    out_flatten = output.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum(np.abs(out_flatten - expect_flatten) /
                     np.abs(expect_flatten) > limit).astype(np.int32)
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        #print("[FAILED]", "err_cnt = ", err_cnt, "/", limit_cnt)
        return False

    #print("[SUCCESS]", "err_cnt = ", err_cnt, "/", limit_cnt)
    return True


def gen_ms_tensor(input_np_list, mstype):
    input_tensor_list = []
    for input_np in input_np_list:
        input_tensor_list.append(Tensor(input_np, dtype=mstype))
    return input_tensor_list


def run_expect_single(x_np, y_np, b_np=0, trans_a=False, trans_b=True):
    if (not trans_a) and (not trans_b):
        expect = np.matmul(x_np, y_np) + b_np
    elif (not trans_a) and trans_b:
        expect = np.matmul(x_np, y_np.T) + b_np
    elif trans_a and (not trans_b):
        expect = np.matmul(x_np.T, y_np) + b_np
    else:  # trans_a and trans_b
        expect = np.matmul(x_np.T, y_np.T) + b_np
    return expect
