# Copyright 2021 Huawei Technologies Co., Ltd
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
"""PoseNet"""
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.context import ParallelMode
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

def weight_variable():
    """Weight variable."""
    return TruncatedNormal(0.02)

class Conv2dBlock(nn.Cell):
    """
     Basic convolutional block
     Args:
         in_channles (int): Input channel.
         out_channels (int): Output channel.
         kernel_size (int): Input kernel size. Default: 1
         stride (int): Stride size for the first convolutional layer. Default: 1.
         padding (int): Implicit paddings on both sides of the input. Default: 0.
         pad_mode (str): Padding mode. Optional values are "same", "valid", "pad". Default: "same".
      Returns:
          Tensor, output tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="same"):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode, weight_init=weight_variable())
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception(nn.Cell):
    """
    Inception Block
    """
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = Conv2dBlock(in_channels, n1x1, kernel_size=1)
        self.b2 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red, kernel_size=1),
                                     Conv2dBlock(n3x3red, n3x3, kernel_size=3, padding=0)])
        # kernel_size = 3: depend on googlenet
        self.b3 = nn.SequentialCell([Conv2dBlock(in_channels, n5x5red, kernel_size=1),
                                     Conv2dBlock(n5x5red, n5x5, kernel_size=3, padding=0)])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode="same")
        self.b4 = Conv2dBlock(in_channels, pool_planes, kernel_size=1)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """construct"""
        branch1 = self.b1(x)
        branch2 = self.b2(x)
        branch3 = self.b3(x)
        cell = self.maxpool(x)
        branch4 = self.b4(cell)
        return self.concat((branch1, branch2, branch3, branch4))

class PoseNet(nn.Cell):
    """
    PoseNet architecture
    """
    def __init__(self):
        super(PoseNet, self).__init__()
        self.conv1 = Conv2dBlock(3, 64, kernel_size=7, stride=2, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.conv2 = Conv2dBlock(64, 64, kernel_size=1)
        self.conv3 = Conv2dBlock(64, 192, kernel_size=3, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.block3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.block3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.block4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.block4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.block4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.block4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.block4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")

        self.block5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.block5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool5x5 = nn.AvgPool2d(kernel_size=5, stride=3, pad_mode="valid")
        self.conv1x1_1 = Conv2dBlock(512, 128, kernel_size=1, stride=1)
        self.conv1x1_2 = Conv2dBlock(528, 128, kernel_size=1, stride=1)
        self.fc2048 = nn.Dense(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout7 = nn.Dropout(0.7)
        self.cls_fc_pose_xyz_1024 = nn.Dense(1024, 512)
        self.cls_fc_pose_xyz_512 = nn.Dense(512, 3)
        self.cls_fc_pose_wpqr_1024 = nn.Dense(1024, 4)

        self.avgpool7x7 = nn.AvgPool2d(kernel_size=7, stride=1, pad_mode="valid")
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(1024, 2048)
        self.dropout5 = nn.Dropout(0.5)
        self.cls_fc_pose_xyz = nn.Dense(2048, 3)
        self.cls_fc_pose_wpqr = nn.Dense(2048, 4)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.block3a(x)
        x = self.block3b(x)
        x = self.maxpool3(x)
        x = self.block4a(x)

        cls1 = self.avgpool5x5(x)
        cls1 = self.conv1x1_1(cls1)
        cls1 = self.relu(cls1)
        cls1 = self.flatten(cls1)
        cls1 = self.fc2048(cls1)
        cls1 = self.relu(cls1)
        cls1 = self.dropout7(cls1)
        cls1_fc_pose_xyz = self.cls_fc_pose_xyz_1024(cls1)
        cls1_fc_pose_xyz = self.cls_fc_pose_xyz_512(cls1_fc_pose_xyz)
        cls1_fc_pose_wpqr = self.cls_fc_pose_wpqr_1024(cls1)

        x = self.block4b(x)
        x = self.block4c(x)
        x = self.block4d(x)

        cls2 = self.avgpool5x5(x)
        cls2 = self.conv1x1_2(cls2)
        cls2 = self.relu(cls2)
        cls2 = self.flatten(cls2)
        cls2 = self.fc2048(cls2)
        cls2 = self.relu(cls2)
        cls2 = self.dropout7(cls2)
        cls2_fc_pose_xyz = self.cls_fc_pose_xyz_1024(cls2)
        cls2_fc_pose_xyz = self.cls_fc_pose_xyz_512(cls2_fc_pose_xyz)
        cls2_fc_pose_wpqr = self.cls_fc_pose_wpqr_1024(cls2)

        x = self.block4e(x)
        x = self.maxpool4(x)
        x = self.block5a(x)
        x = self.block5b(x)

        cls3 = self.dropout5(x)
        cls3 = self.avgpool7x7(cls3)
        cls3 = self.flatten(cls3)
        cls3 = self.fc(cls3)
        cls3 = self.relu(cls3)
        cls3 = self.dropout5(cls3)
        cls3_fc_pose_xyz = self.cls_fc_pose_xyz(cls3)
        cls3_fc_pose_wpqr = self.cls_fc_pose_wpqr(cls3)

        return cls1_fc_pose_xyz, cls1_fc_pose_wpqr, \
               cls2_fc_pose_xyz, cls2_fc_pose_wpqr, \
               cls3_fc_pose_xyz, cls3_fc_pose_wpqr


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 50.0

clip_grad = C.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients
    Inputs:
        clip_type: The way to clip, 0 for 'value', 1 for 'norm'
        clip_value: Specifies how much to clip
        grad: Gradients
    Outputs:
        tuple[Tensor], clipped gradients
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad

class PoseTrainOneStepCell(nn.Cell):
    r"""
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Union[Cell]): Optimizer for updating the weights.
        sens (numbers.Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If `sens` is not a number.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(PoseTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.hyper_map = C.HyperMap()
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss
