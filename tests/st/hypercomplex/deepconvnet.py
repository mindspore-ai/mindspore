from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
import mindspore.hypercomplex.dual as ops


class DeepConvNet(nn.Cell):
    def __init__(self):
        super(DeepConvNet, self).__init__()

        self.conv1 = ops.Conv1d(1, 16, kernel_size=6, stride=2, padding=2, pad_mode='pad')
        self.bn1 = ops.BatchNorm1d(16)
        self.avg_pool1 = ops.AvgPool1d(kernel_size=2, stride=2)
        self.pad1 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 2)), mode='CONSTANT')

        self.conv2 = ops.Conv1d(16, 32, kernel_size=3, stride=2, padding=0)
        self.bn2 = ops.BatchNorm1d(32)
        self.avg_pool2 = ops.AvgPool1d(kernel_size=2, stride=2)

        self.conv3 = ops.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn3 = ops.BatchNorm1d(64)
        self.avg_pool3 = ops.AvgPool1d(kernel_size=2, stride=2)

        self.conv4 = ops.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn4 = ops.BatchNorm1d(64)
        self.avg_pool4 = ops.AvgPool1d(kernel_size=2, stride=2)

        self.conv5 = ops.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.conv6 = ops.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn6 = ops.BatchNorm1d(128)
        self.avg_pool6 = ops.AvgPool1d(kernel_size=2, stride=2)

        self.shape_op = P.Shape()
        self.reshape = P.Reshape()
        self.permute = P.Transpose()
        self.flatten = P.Flatten()

        self.fc1 = ops.Dense(4096, 1024)
        self.fc2 = nn.Dense(2048, 84)

        self.relu = ops.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, u: Tensor) -> Tensor:
        u = self.conv1(u)
        u = self.bn1(u)
        u = self.relu(u)
        u = self.avg_pool1(u)
        u = self.pad1(u)

        u = self.conv2(u)
        u = self.bn2(u)
        u = self.relu(u)
        u = self.avg_pool2(u)

        u = self.conv3(u)
        u = self.bn3(u)
        u = self.relu(u)
        u = self.avg_pool3(u)

        u = self.conv4(u)
        u = self.bn4(u)
        u = self.relu(u)
        u = self.avg_pool4(u)

        u = self.conv5(u)
        u = self.relu(u)

        u = self.conv6(u)
        u = self.bn6(u)
        u = self.relu(u)
        u = self.avg_pool6(u)

        u_shape = self.shape_op(u)
        u = self.reshape(u, (u_shape[0], u_shape[1], -1))
        u = self.fc1(u)
        u = self.relu(u)

        u = self.permute(u, (1, 0, 2))
        x = self.flatten(u)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
