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
    context.set_auto_parallel_context(device_num=4, global_rank=3, mirror_mean=True, cast_before_mirror=False,
                                      parallel_mode="auto_parallel", parameter_broadcast=False)
    device_num = context.get_auto_parallel_context("device_num")
    global_rank = context.get_auto_parallel_context("global_rank")
    mirror_mean = context.get_auto_parallel_context("mirror_mean")
    cast_before_mirror = context.get_auto_parallel_context("cast_before_mirror")
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    parameter_broadcast = context.get_auto_parallel_context("parameter_broadcast")
    assert device_num == 4
    assert global_rank == 3
    assert mirror_mean
    assert not cast_before_mirror
    assert parallel_mode == "auto_parallel"
    assert not parameter_broadcast

    auto_parallel_context().set_communication_backend("hccl")
    backend = auto_parallel_context().get_communication_backend()
    assert backend == "hccl"

    auto_parallel_context().set_device_num(4)
    device_num = auto_parallel_context().get_device_num()
    device_num_is_set = auto_parallel_context().get_device_num_is_set()
    assert device_num == 4
    assert device_num_is_set

    auto_parallel_context().set_global_rank(4)
    global_rank = auto_parallel_context().get_global_rank()
    assert global_rank == 4

    auto_parallel_context().set_mirror_mean(True)
    mirror_mean = auto_parallel_context().get_mirror_mean()
    assert mirror_mean

    auto_parallel_context().set_cast_before_mirror(False)
    cast_before_mirror = auto_parallel_context().get_cast_before_mirror()
    assert not cast_before_mirror

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

    auto_parallel_context().set_enable_parallel_optimizer(True)
    assert auto_parallel_context().get_enable_parallel_optimizer() is True
    assert not auto_parallel_context().get_all_reduce_fusion_split_indices()


def test_reset_auto_parallel_context():
    context.reset_auto_parallel_context()
    device_num = context.get_auto_parallel_context("device_num")
    global_rank = context.get_auto_parallel_context("global_rank")
    mirror_mean = context.get_auto_parallel_context("mirror_mean")
    cast_before_mirror = context.get_auto_parallel_context("cast_before_mirror")
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    parameter_broadcast = context.get_auto_parallel_context("parameter_broadcast")
    device_num_is_set = auto_parallel_context().get_device_num_is_set()
    parameter_broadcast_is_set = auto_parallel_context().get_parameter_broadcast_is_set()
    assert device_num == 1
    assert global_rank == 0
    assert not mirror_mean
    assert cast_before_mirror
    assert parallel_mode == "stand_alone"
    assert not parameter_broadcast
    assert not device_num_is_set
    assert not parameter_broadcast_is_set
