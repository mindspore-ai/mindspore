# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Group Loss Scale Manager"""
from __future__ import absolute_import
from __future__ import division

from mindspore.nn.cell import Cell
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple


__all__ = ["GroupLossScaleManager"]


class GroupLossScaleManager(Cell):
    r"""
    Enhanced hybrid precision algorithm supports multi-layer application of different loss scales and
    dynamic updating of loss scales.

    Args:
        init_loss_scale (Number): The initialized loss scale value.
        loss_scale_groups (List): The loss scale groups, which are divided from the param list.

    Inputs:
        - **x** (Tensor) - The output of last operator.
        - **layer1** (Int) - Current network layer value.
        - **layer2** (Int) - Last network layer value.

    Outputs:
        - **out** (Tensor) - A tensor with a group of loss scale tags that marks
          the loss scale group number of the current tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import boost, nn
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, enhanced_amp, num_class=10, num_channel=1):
        ...         super(Net, self).__init__()
        ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        ...         self.fc1 = nn.Dense(16*5*5, 120, weight_init='ones')
        ...         self.fc2 = nn.Dense(120, 84, weight_init='ones')
        ...         self.fc3 = nn.Dense(84, num_class, weight_init='ones')
        ...         self.relu = nn.ReLU()
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.flatten = nn.Flatten()
        ...         self.enhanced_amp = enhanced_amp
        ...
        ...     def construct(self, x):
        ...         x = self.enhanced_amp(x, 0, 1)
        ...         x = self.max_pool2d(self.relu(self.conv1(x)))
        ...         x = self.max_pool2d(self.relu(self.conv2(x)))
        ...         x = self.flatten(x)
        ...         x = self.enhanced_amp(x, 1, 2)
        ...         x = self.relu(self.fc1(x))
        ...         x = self.relu(self.fc2(x))
        ...         x = self.fc3(x)
        ...         x = self.enhanced_amp(x, 2, 3)
        ...         return x
        >>>
        >>> loss_scale_manager = boost.GroupLossScaleManager(4096, [])
        >>> net = Net(loss_scale_manager)
        >>> param_group1 = []
        >>> param_group2 = []
        >>> for param in net.trainable_params():
        ...     if 'conv' in param.name:
        ...         param_group1.append(param)
        ...     else:
        ...         param_group2.append(param)
        >>> loss_scale_manager.loss_scale_groups = [param_group1, param_group2]
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> boost_config_dict = {"boost": {"mode": "manual", "less_bn": False, "grad_freeze": False, "adasum": False,
        ...                      "grad_accumulation": False, "dim_reduce": False, "loss_scale_group": True}}
        >>> model = ms.train.Model(net, loss_fn=loss, optimizer=optim, metrics=None,
        ...                        loss_scale_manager=loss_scale_manager,
        ...                        boost_level="O1", boost_config_dict=boost_config_dict)
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>> model.train(2, dataset)
    """
    def __init__(self, init_loss_scale, loss_scale_groups):
        super(GroupLossScaleManager, self).__init__()
        self._loss_scale = init_loss_scale
        self.loss_scale_groups = loss_scale_groups
        self.loss_scale_number = 0
        self.layer_loss_scale = None
        self.dynamic_loss_scale = None

    def set_loss_scale_status(self, loss_scale_number, init_loss_scale):
        """
        Generate dynamic loss scale tuple and set overflow status list.

        Args:
            loss_scale_number (int): The number of loss scale.
            init_loss_scale (float): The initialized loss scale.
        """
        self.loss_scale_number = loss_scale_number
        inner_list = [P._DynamicLossScale(layer=x) for x in range(loss_scale_number + 1)] # pylint: disable=W0212
        self.layer_loss_scale = tuple(inner_list)
        self.dynamic_loss_scale = ParameterTuple(Parameter(Tensor(1, mstype.float32),
                                                           name='layer_loss_scale_{}'.format(x), requires_grad=False)
                                                 for x in range(loss_scale_number + 2))
        if isinstance(init_loss_scale, list):
            for i, value in enumerate(init_loss_scale):
                self.dynamic_loss_scale[i + 1].set_data(value)
        else:
            for i in range(self.loss_scale_number):
                self.dynamic_loss_scale[i + 1].set_data(init_loss_scale)

    def update_loss_scale_status(self, layer, update_ratio):
        """
        Update dynamic loss scale.

        Args:
            layer (int): Current layer.
            update_ratio (float): The ratio of loss scale update.

        Outputs:
            float, new loss scale.
        """
        layer = layer + 1
        new_loss_scale = self.dynamic_loss_scale[layer] * update_ratio
        P.Assign()(self.dynamic_loss_scale[layer], new_loss_scale)
        return new_loss_scale

    def construct(self, x, layer1, layer2):
        x = self.layer_loss_scale[layer1](x, self.dynamic_loss_scale[layer1] / self.dynamic_loss_scale[layer2])
        return x

    def get_loss_scale(self):
        """
        Get loss scale value.

        Returns:
            bool, `loss_scale` value.
        """
        return self._loss_scale

    def get_update_cell(self):
        """
        Returns the instance of :class:`mindspore.boost.GroupLossScaleManager`.

        Returns:
            :class:`mindspore.boost.GroupLossScaleManager`.
        """
        return self
