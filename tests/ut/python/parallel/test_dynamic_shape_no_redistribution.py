# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter, ops
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy1=None, strategy2=None, strategy3=None, strategy4=None):
        super().__init__()
        self.slice = P.StridedSlice().shard(strategy1)
        self.gather = P.Gather().shard(strategy2)
        self.reshape = P.Reshape()
        self.begin = (0, 0)
        self.strides = (1, 1)
        self.gather_w = Parameter(Tensor(np.ones([8, 16]), dtype=ms.float32), "w1")
        self.matmul1 = P.MatMul().shard(strategy3)
        self.matmul2 = P.MatMul().shard(strategy4)
        self.matmul1_w = Parameter(Tensor(np.ones([16, 64]), dtype=ms.float32), "w2")
        self.matmul2_w = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), "w3")

    def construct(self, x):
        shape = P.Shape()(x)[-1]
        shape = shape - 1
        end = (1, shape)
        out = self.slice(x, self.begin, end, self.strides)
        out = self.gather(self.gather_w, out, 0)
        out = self.reshape(out, (-1, 16))
        out = self.matmul1(out, self.matmul1_w)
        out = self.matmul2(out, self.matmul2_w)
        return out


class PadV3Net(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.pad = P.PadV3().shard(strategy2)
        self.weight = Parameter(weight, "w1")
        self.value = Tensor([0])
        self.shape = P.Shape()

    def construct(self, x):
        out = self.add(x, self.weight)
        shape = self.shape(out)[-1]
        shape = 1024 - shape
        out = self.pad(out, (0, shape), self.value)
        return out


class PadV3Net2(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.pad = P.PadV3().shard(strategy2)
        self.weight = Parameter(weight, "w1")
        self.value = Tensor([0])
        self.shape = P.Shape()
        self.paddings = Tensor([0, 0, 0])
        self.s = P.ScalarToTensor()
        self.concat = P.Concat()

    def construct(self, x):
        out = self.add(x, self.weight)
        shape = self.shape(out)[-1]
        shape = Tensor([1024]) - self.s(shape, ms.int32)
        paddings = self.concat((self.paddings, shape))
        out = self.pad(out, paddings, self.value)
        return out


class PadV3Net3(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.pad = P.PadV3().shard(strategy2)
        self.weight = Parameter(weight, "w1")
        self.value = Tensor([0])
        self.shape = P.Shape()
        self.paddings = Tensor([0, 0, 0], ms.int32)
        self.s = P.ScalarToTensor()
        self.concat = P.Concat()

    def construct(self, x):
        out = self.add(x, self.weight)
        shape = P.Sub()(Tensor([1024]), ops.dyn_shape(out)[-1]).reshape((1,)).astype(ms.int32)
        paddings = self.concat((self.paddings, shape))
        out = self.pad(out, paddings, self.value)
        return out


class ReshapeNet(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.relu = P.ReLU().shard(strategy2)

    def construct(self, x):
        out = self.add(x, self.weight)
        shape = self.shape(out)
        out = self.reshape(out, shape)
        out = self.relu(out)
        return out


class NoStrategyNet(Cell):
    def __init__(self, weight):
        super().__init__()
        self.add = P.Add()
        self.weight = Parameter(weight, "w1")
        self.transpose = P.Transpose()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.add(x, self.weight)
        out = self.transpose(out, (0, 2, 1))
        shape = self.shape(out)
        out = self.reshape(out, shape)
        out = self.relu(out)
        return out


def test_shape_sub():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1),)  # stridedslice
    strategy2 = ((1, 1), (1, 1))  # gather
    strategy3 = ((1, 1), (1, 8))  # matmul1
    strategy4 = ((1, 8), (8, 1))  # matmul2
    net = Net(strategy1, strategy2, strategy3, strategy4)
    input_x = Tensor(shape=[1, None], dtype=ms.int32)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('AllReduce-0', ['MatMul-1'])
    assert validator.check_parameter_shape("w2", [16, 8])
    assert validator.check_parameter_shape("w3", [8, 128])


def test_padv3_dynamic():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1), (1, 1, 1))
    strategy2 = ((1, 1, 1),)
    input_x = Tensor(shape=[32, 16, None], dtype=ms.int32)
    weight = Tensor(np.ones([32, 16, 1]), dtype=ms.float32)
    net = PadV3Net(weight, strategy1, strategy2)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])


def test_padv3_paddings_concat_scalar_to_tensor_dynamic():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1), (1, 1, 1))
    strategy2 = ((1, 1, 1),)
    input_x = Tensor(shape=[32, 16, None], dtype=ms.int32)
    weight = Tensor(np.ones([32, 16, 1]), dtype=ms.float32)
    net = PadV3Net2(weight, strategy1, strategy2)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])


def test_padv3_concat_tensor_shape_dynamic():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1), (1, 1, 1))
    strategy2 = ((1, 1, 1),)
    input_x = Tensor(shape=[32, 16, None], dtype=ms.int32)
    weight = Tensor(np.ones([32, 16, 1]), dtype=ms.float32)
    net = PadV3Net3(weight, strategy1, strategy2)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Add-0'])


def test_reshape_input_is_shape():
    """
    Feature: test dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1), (1, 1, 1))
    strategy2 = ((1, 1, 1),)
    input_x = Tensor(shape=[32, 16, None], dtype=ms.int32)
    weight = Tensor(np.ones([32, 16, 1]), dtype=ms.float32)
    net = ReshapeNet(weight, strategy1, strategy2)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-0'])


def test_dynamic_shape_gen_batch_parallel_strategy():
    """
    Feature: test dynamic shape generate batch parallel strategy, and can not use all devices
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    input_x = Tensor(shape=[None, 16, None], dtype=ms.int32)
    weight = Tensor(np.ones([1, 16, 1]), dtype=ms.float32)
    net = NoStrategyNet(weight)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Reshape-0', ['Transpose-0'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-0'])


class GatherNet(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None):
        super().__init__()
        self.gather = P.Gather().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.sqrt = P.Sqrt().shard(strategy2)

    def construct(self, x):
        out = self.gather(self.weight, x, 0)
        out = self.sqrt(out)
        return out


def test_gather_indices_dynamic():
    """
    Feature: test gather, split param, indices dynamic shape
    Description: no redistribution, but has replace op for gather
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((1, 1, 1),)
    weight = Tensor(np.ones([32, 64]), dtype=ms.float32)
    net = GatherNet(weight, strategy1, strategy2)
    input_x = Tensor(shape=[4, None], dtype=ms.int32)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Sqrt-0', ['AllReduce-0'])
    assert validator.check_parameter_shape('w1', [4, 64])


class AttentionNet(Cell):
    def __init__(self, weight, bias, strategy1=None, strategy2=None, strategy3=None, strategy4=None):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.add = P.Add().shard(strategy2)
        self.bias = Parameter(bias, "bias")
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(strategy3)
        self.realdiv = P.RealDiv().shard(strategy4)

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.add(out, self.bias)
        out = self.reshape(out, (1, -1, 16, 4))
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.realdiv(out, 1.0)
        return out


@pytest.mark.skip(reason="offline this testcase for tensor redistribution temporarily, "
                         "online after can tracing ir.")
def test_attention_reshape():
    """
    Feature: test attention parallel, the dst shape of reshape is dynamic
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((1, 8), (8,))
    strategy3 = ((1, 1, 8, 1),)
    strategy4 = ((1, 8, 1, 1), ())
    weight = Tensor(np.ones([32, 64]), dtype=ms.float32)
    bias = Tensor(np.ones([64]), dtype=ms.float32)
    net = AttentionNet(weight, bias, strategy1, strategy2, strategy3, strategy4)
    input_x = Tensor(shape=[None, 32], dtype=ms.float32)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0'])
    assert validator.check_node_inputs_has('Transpose-0', ['Reshape-0'])
    assert validator.check_parameter_shape('w1', [32, 8])
    assert validator.check_parameter_shape('bias', [8])
    reshape_expect_inputs = ['Add-0', '((1, -1, 2, 4))']
    assert validator.check_node_inputs_fuzzy_match('Reshape-0', reshape_expect_inputs)


class AttentionNet2(Cell):
    def __init__(self, weight, bias, strategy1=None, strategy2=None, strategy3=None, strategy4=None):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.add = P.Add().shard(strategy2)
        self.bias = Parameter(bias, "bias")
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(strategy3)
        self.realdiv = P.RealDiv().shard(strategy4)

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.add(out, self.bias)
        bs = P.Shape()(out)[0]
        out = self.reshape(out, (bs // 2, 2, 16, 4))
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.realdiv(out, 1.0)
        return out


def test_attention_reshape_dst_shape_is_not_value_node():
    """
    Feature: test attention parallel, the dst shape of reshape is not value node, only one dim is dynamic for reshape
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((1, 8), (8,))
    strategy3 = ((1, 1, 8, 1),)
    strategy4 = ((1, 8, 1, 1), ())
    weight = Tensor(np.ones([32, 64]), dtype=ms.float32)
    bias = Tensor(np.ones([64]), dtype=ms.float32)
    net = AttentionNet2(weight, bias, strategy1, strategy2, strategy3, strategy4)
    input_x = Tensor(shape=[None, 32], dtype=ms.float32)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0'])
    assert validator.check_node_inputs_has('Transpose-0', ['Reshape-0'])
    assert validator.check_parameter_shape('w1', [32, 8])
    assert validator.check_parameter_shape('bias', [8])


class StridedSliceReshapeNet(Cell):
    def __init__(self, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.slice = P.StridedSlice().shard(strategy1).add_prim_attr("skip_redistribution", True)
        self.gather = P.Gather().shard(strategy2)
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.begin = (0, 0)
        self.strides = (1, 1)
        self.gather_w = Parameter(Tensor(np.ones([8, 16]), dtype=ms.float32), "w1")
        self.mul = P.Mul().shard(strategy3)
        self.mul_w = Parameter(Tensor(np.ones([32, 1]), dtype=ms.float32), "w2")
        self.shape = P.Shape()

    def construct(self, x):
        shape = P.Shape()(x)[-1]
        end = (32, shape)
        out = self.slice(x, self.begin, end, self.strides)
        out = self.gather(self.gather_w, out, 0)
        s = self.shape(out)
        out = self.reshape(out, (32, s[1] * s[2]))
        out = self.mul(out, self.mul_w)
        return out


def test_modify_inputs_of_stridedslice_and_reshape():
    """
    Feature: test modify inputs
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0, full_batch=False)
    strategy1 = ((8, 1),)
    strategy2 = ((1, 1), (8, 1))
    strategy3 = ((8, 1), (8, 1))
    net = StridedSliceReshapeNet(strategy1, strategy2, strategy3)
    input_x = Tensor(shape=[32, None], dtype=ms.int32)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('MakeTuple-1', [4])
    assert validator.check_node_inputs_has('MakeTuple-2', [4])


class ConcatStridedSliceNet(Cell):
    def __init__(self, strategy1=None, strategy2=None):
        super().__init__()
        self.slice = P.StridedSlice().shard(strategy2)
        self.concat = P.Concat().shard(strategy1)
        self.begin = (0, 0)
        self.strides = (1, 1)
        self.end_1 = Tensor([4], dtype=ms.int64)
        self.end_2 = Tensor([4], dtype=ms.int64)
        self.relu = P.ReLU()

    def construct(self, x):
        end_1 = self.relu(self.end_1)
        end_2 = self.relu(self.end_2)
        end = self.concat((end_1, end_2))
        out = self.slice(x, self.begin, end, self.strides)
        return out


def test_concat_is_the_input_of_stridedslice():
    """
    Feature: test concat is the input of stridedslice
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0, full_batch=True)
    strategy1 = ((1,), (1,))
    strategy2 = ((1, 1),)
    net = ConcatStridedSliceNet(strategy1, strategy2)
    input_x = Tensor(np.ones([64, 64]), dtype=ms.float32)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('StridedSlice-0', ['TensorToTuple-0'])


class ConcatPadV3Net(Cell):
    def __init__(self, strategy1=None, strategy2=None):
        super().__init__()
        self.pad = P.PadV3().shard(strategy2)
        self.concat = P.Concat().shard(strategy1)
        self.pad_1 = Tensor([0], dtype=ms.int64)
        self.pad_2 = Tensor([0], dtype=ms.int64)
        self.relu = P.ReLU()

    def construct(self, x):
        pad_1 = self.relu(self.pad_1)
        pad_2 = self.relu(self.pad_2)
        paddings = self.concat((pad_1, pad_2))
        out = self.pad(x, paddings, 0)
        return out


def test_concat_is_the_input_of_padv3():
    """
    Feature: test concat is the input of padv3
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0,
                                      dataset_strategy=((1, 8, 1, 1),))
    strategy1 = ((1,), (1,))
    strategy2 = ((1, 8, 1, 1), (1,), ())
    net = ConcatPadV3Net(strategy1, strategy2)

    input_x = Tensor(shape=[1, 32, None, 128], dtype=ms.float32)
    net.set_inputs(input_x)
    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('PadV3-0', ['Concat-0'])


def test_dynamic_fillv2():
    """
    Feature: test dynamic fillv2
    Description: no redistribution
    Expectation: compile success
    """

    class DynamicFillNet(Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.fill = P.FillV2().shard(strategy1)
            self.relu = P.ReLU().shard(strategy2)
            self.v = Tensor(1, dtype=ms.float32)
            self.add = P.Add().shard(strategy3)

        def construct(self, x):
            out1 = self.relu(x)
            s = P.Shape()(out1)[0]
            out2 = self.fill((s, 16), self.v)
            out = self.add(out1, out2)
            return out

    strategy1 = ((1, 8), ())
    strategy2 = ((1, 8),)
    strategy3 = ((1, 8), (1, 8))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0,
                                      dataset_strategy=strategy2)
    context.set_context(save_graphs=True)

    x = Tensor(shape=[None, 2], dtype=ms.float32)
    net = DynamicFillNet(strategy1, strategy2, strategy3)
    net.set_inputs(x)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('MakeTuple-1', [2])


def test_dynamic_tile():
    """
    Feature: test dynamic tile
    Description: no redistribution
    Expectation: compile success
    """

    class DynamicTileNet(Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.tile = P.Tile().shard(strategy1)
            self.relu1 = P.ReLU().shard(strategy2)
            self.relu2 = P.ReLU().shard(strategy1)

        def construct(self, x):
            out = self.relu1(x)
            s = P.Shape()(out)[0]
            out = self.tile(out, (1, 16, s, 1, 1))
            out = self.relu2(out)
            return out

    strategy1 = ((1, 8, 1, 1, 1),)
    strategy2 = ((1, 1, 1, 1, 1),)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, dataset_strategy=strategy2)
    context.set_context(save_graphs=True)

    net = DynamicTileNet(strategy1, strategy2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[None, 8, 1, None, 128], dtype=ms.float32)
    net.set_inputs(x)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('MakeTuple-1', [2])


def test_dynamic_mul_broadcast():
    """
    Feature: test dynamic mul broadcast
    Description: no redistribution
    Expectation: compile success
    """

    class DynamicMulNet(Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            self.relu = P.ReLU().shard(strategy2)

        def construct(self, x, y):
            out = self.mul(x, y)
            out = self.relu(out)
            return out

    strategy1 = ((1, 8, 1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 8, 1, 1),)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, dataset_strategy=strategy1)
    context.set_context(save_graphs=True)

    net = DynamicMulNet(strategy1, strategy2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(shape=[None, 64, None, 128], dtype=ms.float32)
    y = Tensor(shape=[None, None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)
    phase = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('ReLU-0', ['Mul-0'])
