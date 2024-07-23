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
from mindspore import context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_lccl_allreduce():
    """
    Feature: lccl operator test.
    Description: msrun lccl all_reduce 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_allreduce.py")
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='allcards', essential_mark='unessential')
def test_lccl_allgather():
    """
    Feature: lccl operator test.
    Description: msrun lccl all_gather 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_allgather.py")
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='allcards', essential_mark='unessential')
def test_lccl_reducescatter():
    """
    Feature: lccl operator test.
    Description: msrun lccl reduce_scatter 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True "
                            "pytest -s test_lccl_reduce_scatter.py")
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='allcards', essential_mark='unessential')
def test_lccl_broadcast():
    """
    Feature: lccl operator test.
    Description: msrun lccl broadcast 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_broadcast.py")
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='allcards', essential_mark='unessential')
def test_lccl_matmul_allreduce():
    """
    Feature: lccl MatMulAllReduce fustion operator test.
    Description: lccl MatMulAllReduce 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_matmul_allreduce.py")
    assert return_code == 0
