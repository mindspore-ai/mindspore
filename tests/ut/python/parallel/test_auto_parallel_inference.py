import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn import Momentum
from mindspore.parallel._cost_model_context import set_cost_model_context


class Net(nn.Cell):
    def __init__(self, input_ch, out_ch):
        super(Net, self).__init__()
        self.dense = nn.Dense(input_ch, out_ch)
        self.relu = P.ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x


def test_inference_phase():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    set_cost_model_context(run_phase=1)

    net = Net(512, 128)
    predict = Tensor(np.ones([64, 512]).astype(np.float32) * 0.001)
    label = Tensor(np.ones([64, 128]).astype(np.float32))

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()
    train_network.set_auto_parallel()

    output = train_network(predict, label)
