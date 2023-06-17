import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import Normal


class LeNet5(nn.Cell):

    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        '''
        Forward network.
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_save_checkpoint(mode):
    """
    Feature: mindspore.save_checkpoint
    Description: Save checkpoint to a specified file.
    Expectation: success
    """
    context.set_context(mode=mode)
    net = LeNet5()
    ms.save_checkpoint(net, "./lenet.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict = ms.load_checkpoint("./lenet.ckpt")
    assert 'conv2.weight' in output_param_dict
    assert 'conv1.weight' not in output_param_dict
    assert 'fc1.bias' not in output_param_dict
