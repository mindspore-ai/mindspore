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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.auto_generate import WeightQuantBatchMatmul
from parallel.utils.utils import ParallelValidator, compile_net

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class WeightQuantBatchMatmulNet(Cell):
    def __init__(self, transpose_x, transpose_weight, strategy):
        super().__init__()
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight).shard(strategy)

    def construct(self, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias):
        out = self.weight_qbmm(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
        return out


def test_weight_quant_batch_matmul_case0():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1), (1, 1), (1,), (1,), (1,), (1,), (1,))

    net = WeightQuantBatchMatmulNet(False, False, strategy)

    x = Parameter(Tensor(np.ones([2048, 1024]), dtype=ms.float16), "x")
    weight = Parameter(Tensor(np.ones([1024, 128]), dtype=ms.int8), "weight")
    antiquant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_scale")
    antiquant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_offset")
    quant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "quant_scale")
    quant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "quant_offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "bias")
    net.set_inputs(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)

    phase = compile_net(net, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [512, 1024])
    assert validator.check_parameter_shape("weight", [1024, 128])
    assert validator.check_parameter_shape("antiquant_scale", [128])
    assert validator.check_parameter_shape("antiquant_offset", [128])
    assert validator.check_parameter_shape("quant_scale", [128])
    assert validator.check_parameter_shape("quant_offset", [128])
    assert validator.check_parameter_shape("bias", [128])


def test_weight_quant_batch_matmul_case1_batch_cut_batch():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1, 1), (4, 1, 1), (1,), (1,), (1,), (1,), (1,))

    net = WeightQuantBatchMatmulNet(False, False, strategy)

    x = Parameter(Tensor(np.ones([32, 2048, 1024]), dtype=ms.float16), "x")
    weight = Parameter(Tensor(np.ones([32, 1024, 128]), dtype=ms.int8), "weight")
    antiquant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_scale")
    antiquant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_offset")
    quant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "quant_scale")
    quant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "quant_offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "bias")
    net.set_inputs(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)

    phase = compile_net(net, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [8, 2048, 1024])
    assert validator.check_parameter_shape("weight", [8, 1024, 128])
    assert validator.check_parameter_shape("antiquant_scale", [128])
    assert validator.check_parameter_shape("antiquant_offset", [128])
    assert validator.check_parameter_shape("quant_scale", [128])
    assert validator.check_parameter_shape("quant_offset", [128])
    assert validator.check_parameter_shape("bias", [128])


def test_weight_quant_batch_matmul_case2_batch_transpose_cut_batch():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1, 1), (1, 1), (1,), (1,), (1,), (1,), (1,))

    net = WeightQuantBatchMatmulNet(True, True, strategy)

    x = Parameter(Tensor(np.ones([32, 1024, 2048]), dtype=ms.float16), "x")
    weight = Parameter(Tensor(np.ones([128, 1024]), dtype=ms.int8), "weight")
    antiquant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_scale")
    antiquant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_offset")
    quant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "quant_scale")
    quant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "quant_offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "bias")
    net.set_inputs(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)

    phase = compile_net(net, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [8, 1024, 2048])
    assert validator.check_parameter_shape("weight", [128, 1024])
    assert validator.check_parameter_shape("antiquant_scale", [128])
    assert validator.check_parameter_shape("antiquant_offset", [128])
    assert validator.check_parameter_shape("quant_scale", [128])
    assert validator.check_parameter_shape("quant_offset", [128])
    assert validator.check_parameter_shape("bias", [128])

def test_weight_quant_batch_matmul_case3_transpose_cut_k():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 2), (1, 2), (1,), (1,), (1,), (1,), (1,))

    net = WeightQuantBatchMatmulNet(False, True, strategy)

    x = Parameter(Tensor(np.ones([2048, 1024]), dtype=ms.float16), "x")
    weight = Parameter(Tensor(np.ones([128, 1024]), dtype=ms.int8), "weight")
    antiquant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_scale")
    antiquant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_offset")
    quant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "quant_scale")
    quant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "quant_offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "bias")
    net.set_inputs(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)

    phase = compile_net(net, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [2048, 512])
    assert validator.check_parameter_shape("weight", [128, 512])
    assert validator.check_parameter_shape("antiquant_scale", [128])
    assert validator.check_parameter_shape("antiquant_offset", [128])
    assert validator.check_parameter_shape("quant_scale", [128])
    assert validator.check_parameter_shape("quant_offset", [128])
    assert validator.check_parameter_shape("bias", [128])


def test_weight_quant_batch_matmul_case4_transpose_cut_n():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 1), (1, 4), (4,), (4,), (4,), (4,), (4,))

    net = WeightQuantBatchMatmulNet(True, False, strategy)

    x = Parameter(Tensor(np.ones([1024, 2048]), dtype=ms.float16), "x")
    weight = Parameter(Tensor(np.ones([1024, 128]), dtype=ms.int8), "weight")
    antiquant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_scale")
    antiquant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_offset")
    quant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "quant_scale")
    quant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "quant_offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "bias")
    net.set_inputs(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)

    phase = compile_net(net, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [1024, 2048])
    assert validator.check_parameter_shape("weight", [1024, 32])
    assert validator.check_parameter_shape("antiquant_scale", [32])
    assert validator.check_parameter_shape("antiquant_offset", [32])
    assert validator.check_parameter_shape("quant_scale", [32])
    assert validator.check_parameter_shape("quant_offset", [32])
    assert validator.check_parameter_shape("bias", [32])


def test_weight_quant_batch_matmul_case5_batch_transpose_cut_k():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 1, 2), (1, 1, 2), (1,), (1,), (1,), (1,), (1,))

    net = WeightQuantBatchMatmulNet(False, True, strategy)

    x = Parameter(Tensor(np.ones([32, 2048, 1024]), dtype=ms.float16), "x")
    weight = Parameter(Tensor(np.ones([32, 128, 1024]), dtype=ms.int8), "weight")
    antiquant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_scale")
    antiquant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "antiquant_offset")
    quant_scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "quant_scale")
    quant_offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "quant_offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.float16), "bias")
    net.set_inputs(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)

    phase = compile_net(net, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [32, 2048, 512])
    assert validator.check_parameter_shape("weight", [32, 128, 512])
    assert validator.check_parameter_shape("antiquant_scale", [128])
    assert validator.check_parameter_shape("antiquant_offset", [128])
    assert validator.check_parameter_shape("quant_scale", [128])
    assert validator.check_parameter_shape("quant_offset", [128])
    assert validator.check_parameter_shape("bias", [128])
