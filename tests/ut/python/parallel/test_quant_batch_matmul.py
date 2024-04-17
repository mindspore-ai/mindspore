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
from mindspore.ops.auto_generate import QuantBatchMatmul
from parallel.utils.utils import ParallelValidator, compile_net

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class QuantBatchMatmulNet(Cell):
    def __init__(self, transpose_x1, transpose_x2, strategy):
        super().__init__()
        self.quant_batch_matmul = QuantBatchMatmul(transpose_x1, transpose_x2).shard(strategy)

    def construct(self, x1, x2, scale, offset, bias):
        out = self.quant_batch_matmul(x1, x2, scale, offset, bias)
        return out


def test_quant_batch_matmul_case1():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1), (1, 1), (1,), (1,), (1,))

    net = QuantBatchMatmulNet(False, False, strategy)

    x1 = Parameter(Tensor(np.ones([2048, 1024]), dtype=ms.int8), "x1")
    x2 = Parameter(Tensor(np.ones([1024, 128]), dtype=ms.int8), "x2")
    scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "scale")
    offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.int32), "bias")
    net.set_inputs(x1, x2, scale, offset, bias)

    phase = compile_net(net, x1, x2, scale, offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x1", [512, 1024])
    assert validator.check_parameter_shape("x2", [1024, 128])
    assert validator.check_parameter_shape("scale", [128])
    assert validator.check_parameter_shape("offset", [128])
    assert validator.check_parameter_shape("bias", [128])


def test_quant_batch_matmul_case2_transpose():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 4), (1, 1), (1,), (1,), (1,))

    net = QuantBatchMatmulNet(True, True, strategy)

    x1 = Parameter(Tensor(np.ones([1024, 2048]), dtype=ms.int8), "x1")
    x2 = Parameter(Tensor(np.ones([128, 1024]), dtype=ms.int8), "x2")
    scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "scale")
    offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.int32), "bias")
    net.set_inputs(x1, x2, scale, offset, bias)

    phase = compile_net(net, x1, x2, scale, offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x1", [1024, 512])
    assert validator.check_parameter_shape("x2", [128, 1024])
    assert validator.check_parameter_shape("scale", [128])
    assert validator.check_parameter_shape("offset", [128])
    assert validator.check_parameter_shape("bias", [128])



def test_quant_batch_matmul_case3_batch1():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1, 1), (4, 1, 1), (1,), (1,), (1,))

    net = QuantBatchMatmulNet(False, False, strategy)

    x1 = Parameter(Tensor(np.ones([32, 2048, 1024]), dtype=ms.int8), "x1")
    x2 = Parameter(Tensor(np.ones([32, 1024, 128]), dtype=ms.int8), "x2")
    scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "scale")
    offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.int32), "bias")
    net.set_inputs(x1, x2, scale, offset, bias)

    phase = compile_net(net, x1, x2, scale, offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x1", [8, 2048, 1024])
    assert validator.check_parameter_shape("x2", [8, 1024, 128])
    assert validator.check_parameter_shape("scale", [128])
    assert validator.check_parameter_shape("offset", [128])
    assert validator.check_parameter_shape("bias", [128])

def test_quant_batch_matmul_case3_batch2():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 1), (4, 1, 1), (1,), (1,), (1,))

    net = QuantBatchMatmulNet(False, False, strategy)

    x1 = Parameter(Tensor(np.ones([2048, 1024]), dtype=ms.int8), "x1")
    x2 = Parameter(Tensor(np.ones([32, 1024, 128]), dtype=ms.int8), "x2")
    scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "scale")
    offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "offset")
    bias = Parameter(Tensor(np.ones([128]), dtype=ms.int32), "bias")
    net.set_inputs(x1, x2, scale, offset, bias)

    phase = compile_net(net, x1, x2, scale, offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x1", [2048, 1024])
    assert validator.check_parameter_shape("x2", [8, 1024, 128])
    assert validator.check_parameter_shape("scale", [128])
    assert validator.check_parameter_shape("offset", [128])
    assert validator.check_parameter_shape("bias", [128])

def test_quant_batch_matmul_optional_input_case1():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 1), (4, 1, 1), (1,), (1,))

    net = QuantBatchMatmulNet(False, False, strategy)

    x1 = Parameter(Tensor(np.ones([2048, 1024]), dtype=ms.int8), "x1")
    x2 = Parameter(Tensor(np.ones([32, 1024, 128]), dtype=ms.int8), "x2")
    scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "scale")
    offset = Parameter(Tensor(np.ones([128]), dtype=ms.float32), "offset")
    bias = None
    net.set_inputs(x1, x2, scale, offset, bias)

    phase = compile_net(net, x1, x2, scale, offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x1", [2048, 1024])
    assert validator.check_parameter_shape("x2", [8, 1024, 128])
    assert validator.check_parameter_shape("scale", [128])
    assert validator.check_parameter_shape("offset", [128])

def test_quant_batch_matmul_optional_input_case2():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 1), (4, 1, 1), (1,))

    net = QuantBatchMatmulNet(False, False, strategy)

    x1 = Parameter(Tensor(np.ones([2048, 1024]), dtype=ms.int8), "x1")
    x2 = Parameter(Tensor(np.ones([32, 1024, 128]), dtype=ms.int8), "x2")
    scale = Parameter(Tensor(np.ones([128]), dtype=ms.uint64), "scale")
    offset = None
    bias = None
    net.set_inputs(x1, x2, scale, offset, bias)

    phase = compile_net(net, x1, x2, scale, offset, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x1", [2048, 1024])
    assert validator.check_parameter_shape("x2", [8, 1024, 128])
    assert validator.check_parameter_shape("scale", [128])
