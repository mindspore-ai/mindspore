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
Test module for GPTO execution order.
"""
import os
from tests.mark_utils import arg_mark

@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=0.01
)
def test_gpto_exec_order_interface_static():
    """
    Feature: this test calls gpto_net_static_ops.py
    Description: this test uses msrun to run the gpto interface test on static shape operators
    Expectation: the test should pass without any error
    """

    # Test static interface 1
    fake_profile_data = [
        "Default/Sub-op1,1000",
        "Default/AllReduce-op0,10000",
        "Default/Add-op0,500",
        "Default/Sub-op0,500",
        "Default/Mul-op0,300"
    ]

    with open('fake_profile.csv', 'w', encoding="utf-8") as f:
        for line in fake_profile_data:
            f.write(f"{line}\n")

    os.environ['MS_GPTO_OPTIONS'] = '{"mode":"advance", "passes":"off", "bias":"1.3", "verification":"on", \
                                    "algo":"", "profile_cost_info":"fake_profile.csv"}'
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10967 --join=True gpto_net_static_ops.py"
    )
    del os.environ['MS_GPTO_OPTIONS']
    assert return_code == 0

    # Test static interface 2
    os.environ['MS_GPTO_OPTIONS'] = '{"mode":"basic", "passes":"on", "verification":"off", \
                                    "algo":"SortByReversePostOrder"}'
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10967 --join=True gpto_net_static_ops.py"
    )
    del os.environ['MS_GPTO_OPTIONS']
    assert return_code == 0


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_gpto_exec_order_interface_dynamic():
    """
    Feature: this test calls gpto_net_dynamic_ops.py
    Description: this test uses msrun to run the gpto interface test on dynamic shape operators
    Expectation: the test should pass without any error
    """

    fake_tensors_data = [
        "name,out_tensors_work_tensors",
        "Default/MatMul-op0,25,100",
        "Default/relu-RelU/ReLU-op0,25,"
    ]

    with open('fake_tensors.csv', 'w', encoding="utf-8") as f:
        for line in fake_tensors_data:
            f.write(f"{line}\n")

    os.environ['MS_GPTO_OPTIONS'] = '{"mode":"basic", "passes":"off", "verification":"on", \
                                    "dynamic_tensor_info":"fake_tensors.csv"}'
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10967 --join=True gpto_net_dynamic_ops.py"
    )
    del os.environ['MS_GPTO_OPTIONS']
    assert return_code == 0
    