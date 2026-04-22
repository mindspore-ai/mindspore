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
import os
from tests.mark_utils import arg_mark
import numpy as np


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_dvm_allreduce():
    """
    Feature: DVM operator test.
    Description: msrun dvm allreduce 4P case.
    Expectation: success
    """
    return_code = os.system(
        "MS_DEV_GRAPH_KERNEL_FLAGS='--enable_cluster_ops=AllReduce' "\
        "msrun --worker_num=4 --local_worker_num=4 --join=True --log_dir=./dvm_allreduce_log "\
        "python test_dvm_allreduce.py"
    )
    assert return_code == 0

    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --join=True --log_dir=./hccl_allreduce_log "\
        "python test_dvm_allreduce.py"
    )
    assert return_code == 0

    for i in range(4):
        dvm_res = np.load("./dvm_allreduce_res_" + str(i) + ".npy")
        hccl_res = np.load("./hccl_allreduce_res_" + str(i) + ".npy")
        assert np.allclose(dvm_res, hccl_res, 5e-3, 5e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_dvm_matmul_allreduce():
    """
    Feature: DVM operator test.
    Description: msrun dvm matmul + allreduce 4P case.
    Expectation: success
    """
    return_code = os.system(
        "MS_DEV_GRAPH_KERNEL_FLAGS='--enable_cluster_ops=MatMul,AllReduce' "\
        "msrun --worker_num=4 --local_worker_num=4 --join=True --log_dir=./dvm_matmul_allreduce_log "\
        "python test_dvm_matmul_allreduce.py"
    )
    assert return_code == 0

    return_code = os.system(
        "MS_DEV_GRAPH_KERNEL_FLAGS='--enable_cluster_ops=MatMul' "\
        "msrun --worker_num=4 --local_worker_num=4 --join=True --log_dir=./hccl_matmul_allreduce_log "\
        "python test_dvm_matmul_allreduce.py"
    )
    assert return_code == 0

    for i in range(4):
        dvm_res = np.load("./dvm_matmul_allreduce_res_" + str(i) + ".npy")
        hccl_res = np.load("./dvm_matmul_allreduce_res_" + str(i) + ".npy")
        assert np.allclose(dvm_res, hccl_res, 5e-3, 5e-3)
