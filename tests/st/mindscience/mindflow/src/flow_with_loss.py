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
# ==============================================================================
"""flow with loss"""
from mindspore import ops, jit_class
from .utils import get_loss_metric


class FlowWithLoss:
    """
    Base class of user-defined data-driven flow prediction problems.

    Args:
        model (mindspore.nn.Cell): A training or test model.
        loss_fn (Union[str, Cell]): Loss function. Default: ``"mse"``.

    Raises:
        TypeError: If `modle` or `loss_fn` is not mindspore.nn.Cell.
        NotImplementedError: If the member function `get_loss` is not implemented.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, model, loss_fn='mse'):
        self.model = model
        self.loss_fn = get_loss_metric(loss_fn) if isinstance(loss_fn, str) else loss_fn

    def get_loss(self, inputs, labels):
        """
        Compute the loss of the model.

        Args:
            inputs (Tensor): The input data of model.
            labels (Tensor): True values of the samples.
        """
        raise NotImplementedError


@jit_class
class SteadyFlowWithLoss(FlowWithLoss):
    """
    Base class of user-defined steady data-driven problems.

    Args:
        model (mindspore.nn.Cell): A training or test model.
        loss_fn (Union[str, Cell]): Loss function. Default: ``"mse"``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> import mindspore
        >>> from mindflow.pde import SteadyFlowWithLoss
        >>> from mindflow.loss import RelativeRMSELoss
        ...
        >>> class Net(nn.Cell):
        ...    def __init__(self, num_class=10, num_channel=1):
        ...        super(Net, self).__init__()
        ...        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        ...        self.fc1 = nn.Dense(16*5*5, 120, weight_init='ones')
        ...        self.fc2 = nn.Dense(120, 84, weight_init='ones')
        ...        self.fc3 = nn.Dense(84, num_class, weight_init='ones')
        ...        self.relu = nn.ReLU()
        ...        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...        self.flatten = nn.Flatten()
        ...
        ...    def construct(self, x):
        ...        x = self.max_pool2d(self.relu(self.conv1(x)))
        ...        x = self.max_pool2d(self.relu(self.conv2(x)))
        ...        x = self.flatten(x)
        ...        x = self.relu(self.fc1(x))
        ...        x = self.relu(self.fc2(x))
        ...        x = self.fc3(x)
        ...        return x
        ...
        >>> model = Net()
        >>> problem = SteadyFlowWithLoss(model, loss_fn=RelativeRMSELoss())
        ...
        >>> inputs = Tensor(np.random.randn(32, 1, 32, 32), mindspore.float32)
        >>> label = Tensor(np.random.randn(32, 10), mindspore.float32)
        >>> loss = problem.get_loss(inputs, label)
        >>> print(loss)
        680855.1
    """

    def get_loss(self, inputs, labels):
        """
        Compute the loss of training or test model.

        Args:
            inputs (Tensor): The input data of model.
            labels (Tensor): True values of the samples.

        Returns:
            float, loss value.
        """
        pred = self.model(inputs)
        loss = self.loss_fn(pred, labels)
        return loss


@jit_class
class UnsteadyFlowWithLoss(FlowWithLoss):
    """
    Base class of unsteady user-defined data-driven problems.

    Args:
        model (mindspore.nn.Cell): A training or test model.
        t_in (int): Initial time steps. Default: ``1``.
        t_out (int): Output time steps. Default: ``1``.
        loss_fn (Union[str, Cell]): Loss function. Default: ``"mse"``.
        data_format (str): Data format. Default: ``"NTCHW"``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore
        >>> from mindflow.pde import UnsteadyFlowWithLoss
        >>> from mindflow.cell import FNO2D
        >>> from mindflow.loss import RelativeRMSELoss
        ...
        >>> model = FNO2D(in_channels=1, out_channels=1, resolution=64, modes=12)
        >>> problem = UnsteadyFlowWithLoss(model, loss_fn=RelativeRMSELoss(), data_format='NHWTC')
        >>> inputs = Tensor(np.random.randn(32, 64, 64, 1, 1), mindspore.float32)
        >>> label = Tensor(np.random.randn(32, 64, 64, 1, 1), mindspore.float32)
        >>> loss = problem.get_loss(inputs, label)
        >>> print(loss)
        31.999998
    """

    def __init__(self, model, t_in=1, t_out=1, loss_fn='mse', data_format='NTCHW'):
        super(UnsteadyFlowWithLoss, self).__init__(model, loss_fn)
        self.t_in = t_in
        self.t_out = t_out
        check_param_type(t_in, "t_in", data_type=int, exclude_type=bool)
        check_param_type(t_out, "t_out", data_type=int, exclude_type=bool)
        self.data_format = data_format

    def step(self, inputs):
        """
        Support single or multiple time steps training.

        Args:
            inputs (Tensor): Input dataset with data format is "NTCHW" or "NHWTC".

        Returns:
            List(Tensor), Dataset with data format is "NTCHW" or "NHWTC".
        """
        # change inputs dimension: [bs, t_in, c, x1, x2, ...] -> [bs, t_out, c, x1, x2, ...]
        pred_list = []
        for _ in range(self.t_out):
            inp = self._flatten(inputs)
            pred = self.model(inp)
            if self.data_format == 'NTCHW':
                pred = pred.expand_dims(axis=1)
                pred_list.append(pred)
                if self.t_in > 1:
                    inputs = ops.concat([inputs[:, 1:, ...], pred], axis=1)
                else:
                    inputs = pred
            if self.data_format == 'NHWTC':
                pred = pred.expand_dims(axis=-2)
                pred_list.append(pred)
                if self.t_in > 1:
                    inputs = ops.concat([inputs[..., 1:, :], pred], axis=-2)
                else:
                    inputs = pred

        if self.data_format == 'NTCHW':
            pred_list = ops.concat(pred_list, axis=1)
        if self.data_format == 'NHWTC':
            pred_list = ops.concat(pred_list, axis=-2)
        return pred_list

    def get_loss(self, inputs, labels):
        """
        Compute the loss of training or test model.

        Args:
            inputs (Tensor): Dataset with data format is "NTCHW" or "NHWTC".
            labels (Tensor): True values of the samples.

        Returns:
            float, loss value.
        """
        # the dimension of inputs: [bs, t_in, c, x1, x2, ...]
        # the dimension of labels [bs, t_out, c, x1, x2, ...]
        pred = self.step(inputs)

        if self.data_format == 'NTCHW':
            pred = pred[:, -1, ...]
            labels = labels[:, -1, ...]
        if self.data_format == 'NHWTC':
            pred = pred[..., -1, :]
            labels = labels[..., -1, :]
        return self.loss_fn(pred, labels)

    def _flatten(self, inputs):
        """ flatten """
        # [bs, t_in, c, x1, x2, ...] -> [bs, t_in*c, x1, x2, ...]
        dim = len(inputs.shape) - 3
        if self.data_format == 'NTCHW':
            inputs = ops.transpose(inputs, tuple([0] + list(range(3, dim + 3)) + [1, 2]))

        inp_shape = list(inputs.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)
        inputs = ops.reshape(inputs, tuple(inp_shape))

        if self.data_format == 'NTCHW':
            inputs = ops.transpose(inputs, tuple([0, dim + 1] + list(range(1, dim + 1))))
        return inputs
