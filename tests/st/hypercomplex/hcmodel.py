from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.hypercomplex.utils import get_x_and_y, to_2channel
import mindspore.hypercomplex.dual as ops


class HCModel(nn.Cell):

    def __init__(self):
        super(HCModel, self).__init__()
        self.conv1 = ops.Conv2d(1, 10, kernel_size=3)
        self.bn1 = ops.BatchNorm2d(10)
        self.max_pool = ops.MaxPool2d(2)
        self.relu = ops.ReLU()
        self.fc1 = ops.Dense(7290, 256)
        self.fc2 = nn.Dense(512, 10)
        self.concat = P.Concat(1)

    def construct(self, u: Tensor) -> Tensor:
        u = to_2channel(u[:, :1], u[:, 1:])
        u = self.conv1(u)
        u = self.bn1(u)
        u = self.relu(u)
        u = self.max_pool(u)
        u = u.view(2, u.shape[1], -1)
        u = self.fc1(u)
        u = self.relu(u)
        out_x, out_y = get_x_and_y(u)
        out = self.concat([out_x, out_y])
        out = self.fc2(out)
        return out
