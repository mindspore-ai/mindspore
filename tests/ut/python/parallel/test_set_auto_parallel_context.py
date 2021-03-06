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

import pytest

from mindspore import context
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._auto_parallel_context import auto_parallel_context


def test_set_auto_parallel_context():
    context.set_auto_parallel_context(device_num=4, global_rank=3, gradients_mean=True, gradient_fp32_sync=False,
                                      parallel_mode="auto_parallel", parameter_broadcast=False,
                                      communi_parallel_mode="same_server_group_parallel")
    device_num = context.get_auto_parallel_context("device_num")
    global_rank = context.get_auto_parallel_context("global_rank")
    gradients_mean = context.get_auto_parallel_context("gradients_mean")
    gradient_fp32_sync = context.get_auto_parallel_context("gradient_fp32_sync")
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    parameter_broadcast = context.get_auto_parallel_context("parameter_broadcast")
    communi_parallel_mode = context.get_auto_parallel_context("communi_parallel_mode")
    assert device_num == 4
    assert global_rank == 3
    assert gradients_mean
    assert not gradient_fp32_sync
    assert parallel_mode == "auto_parallel"
    assert not parameter_broadcast
    assert communi_parallel_mode == "same_server_group_parallel"

    auto_parallel_context().set_device_num(4)
    device_num = auto_parallel_context().get_device_num()
    device_num_is_set = auto_parallel_context().get_device_num_is_set()
    assert device_num == 4
    assert device_num_is_set

    auto_parallel_context().set_global_rank(4)
    global_rank = auto_parallel_context().get_global_rank()
    assert global_rank == 4

    auto_parallel_context().set_gradients_mean(True)
    gradients_mean = auto_parallel_context().get_gradients_mean()
    assert gradients_mean

    auto_parallel_context().set_gradient_fp32_sync(False)
    gradient_fp32_sync = auto_parallel_context().get_gradient_fp32_sync()
    assert not gradient_fp32_sync

    parameter_broadcast_is_set = auto_parallel_context().get_parameter_broadcast_is_set()
    assert parameter_broadcast_is_set

    with pytest.raises(ValueError):
        context.set_auto_parallel_context(device_num=0)

    with pytest.raises(ValueError):
        context.set_auto_parallel_context(device_num=4097)

    with pytest.raises(ValueError):
        context.set_auto_parallel_context(global_rank=-1)

    with pytest.raises(ValueError):
        context.set_auto_parallel_context(parallel_mode="wrong_mode")

    with pytest.raises(ValueError):
        context.set_auto_parallel_context(global_rank=4096)

    with pytest.raises(ValueError):
        set_algo_parameters(tensor_slice_align_size=0)

    with pytest.raises(ValueError):
        set_algo_parameters(tensor_slice_align_size=1025)

    with pytest.raises(ValueError):
        context.set_auto_parallel_context(communi_parallel_mode="wrong_mode")

    context.set_auto_parallel_context(enable_parallel_optimizer=True)
    assert context.get_auto_parallel_context("enable_parallel_optimizer")
    assert not auto_parallel_context().get_all_reduce_fusion_split_indices()

def test_pipeline_parallel_context():
    context.set_auto_parallel_context(device_num=8, global_rank=4,
                                      parallel_mode="semi_auto_parallel", pipeline_stages=2)
    stage = auto_parallel_context().get_pipeline_stages()
    assert stage == 2

def test_reset_auto_parallel_context():
    context.reset_auto_parallel_context()
    device_num = context.get_auto_parallel_context("device_num")
    global_rank = context.get_auto_parallel_context("global_rank")
    gradients_mean = context.get_auto_parallel_context("gradients_mean")
    gradient_fp32_sync = context.get_auto_parallel_context("gradient_fp32_sync")
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    parameter_broadcast = context.get_auto_parallel_context("parameter_broadcast")
    device_num_is_set = auto_parallel_context().get_device_num_is_set()
    parameter_broadcast_is_set = auto_parallel_context().get_parameter_broadcast_is_set()
    stage = auto_parallel_context().get_pipeline_stages()
    communi_parallel_mode = context.get_auto_parallel_context("communi_parallel_mode")

    assert device_num == 1
    assert global_rank == 0
    assert not gradients_mean
    assert gradient_fp32_sync
    assert parallel_mode == "stand_alone"
    assert not parameter_broadcast
    assert not device_num_is_set
    assert not parameter_broadcast_is_set
    assert stage == 1
    assert communi_parallel_mode == "all_group_parallel"
