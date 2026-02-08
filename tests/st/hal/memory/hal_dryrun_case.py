import numpy as np
from mindspore import context, Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
import os
from tests.device_utils import set_device

context.set_context(mode=ms.GRAPH_MODE)

class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 32

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


def run_lenet_with_mem_tracker():
    set_device()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    tracker_path = os.path.join(cur_dir, "test_replay_mem_tracker")
    os.environ['MS_ALLOC_CONF'] = "memory_tracker_path:%s,enable_vmm:false" % tracker_path
    lenet = LeNet()
    input_data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    lenet(input_data)

run_lenet_with_mem_tracker()
