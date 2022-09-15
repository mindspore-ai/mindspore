import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore import ParameterTuple
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.train.model import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell

class DatasetLenet():
    def __init__(self, data, label, length=3):
        self.data = data
        self.label = label
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data, self.label

    @staticmethod
    def get_dataset_size():
        return 32

    @staticmethod
    def get_repeat_count():
        return 1

    @staticmethod
    def get_batch_size():
        return 32

    def reset(self):
        self.index = 0

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


class MatMulNet(nn.Cell):
    def __init__(self, strategy1, strategy2, matmul_weight):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)
        self.matmul_weight = Parameter(matmul_weight[0], name='weight1')
        self.matmul_weight2 = Parameter(matmul_weight[1], name='weight2')

    def construct(self, inputs):
        out = self.matmul(inputs, self.matmul_weight)
        out = self.matmul1(out, self.matmul_weight2)
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, matmul_weight):
        super().__init__()
        self.block = nn.CellList()
        self.add = P.TensorAdd()
        self.add_list = []
        self.relu_block = nn.CellList()
        for i in range(2):
            cell = MatMulNet(strategy1, strategy2, matmul_weight[i])
            cell.pipeline_stage = i
            self.block.append(cell)
            relu = nn.ReLU()
            relu.pipeline_stage = i
            self.relu_block.append(relu)
            self.add_list.append(
                Parameter(Tensor(np.full((1, 16), 0.1, dtype=np.float32)), name=f"weight{i}"))
        self.add_tuple = ParameterTuple(self.add_list)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
            x = self.relu_block[i](x)
            x = self.add(x, self.add_tuple[i])
        return x


def test_pipeline_split_stage0():
    """
    Feature:pipeline stage0
    Description:pipeline remove monad nodes
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=16, global_rank=0, pipeline_stages=2, full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    weight = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight, weight2], [weight3, weight4]]
    data = Tensor(np.ones([32, 96]), dtype=ms.float32)
    label = Tensor(np.ones([32, 16]), dtype=ms.float32)
    strategy1 = ((2, 1), (1, 4))
    strategy2 = ((1, 2), (2, 2))
    net = Net(strategy1, strategy2, weight_list)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    net = nn.WithLossCell(net, loss)
    net = PipelineCell(net, 4)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1():
    """
    Feature:pipeline stage1
    Description:pipeline remove monad nodes
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=16, global_rank=8, pipeline_stages=2, full_batch=True)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    weight = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight, weight2], [weight3, weight4]]
    data = Tensor(np.ones([128, 96]), dtype=ms.float32)
    label = Tensor(np.ones([128, 16]), dtype=ms.float32)
    strategy1 = ((2, 1), (1, 4))
    strategy2 = ((1, 2), (2, 2))
    net = Net(strategy1, strategy2, weight_list)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    net = nn.WithLossCell(net, loss)
    net = PipelineCell(net, 4)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
