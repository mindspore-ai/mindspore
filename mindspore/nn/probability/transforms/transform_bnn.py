# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Transform DNN to BNN."""
import mindspore.nn as nn
from ...wrap.cell_wrapper import TrainOneStepCell
from ....nn import optim
from ....nn import layer
from ...probability import bnn_layers
from ..bnn_layers.bnn_cell_wrapper import WithBNNLossCell
from ..bnn_layers.conv_variational import ConvReparam
from ..bnn_layers.dense_variational import DenseReparam

__all__ = ['TransformToBNN']


class TransformToBNN:
    r"""
    Transform Deep Neural Network (DNN) model to Bayesian Neural Network (BNN) model.

    Args:
        trainable_dnn (Cell): A trainable DNN model (backbone) wrapped by TrainOneStepCell.
        dnn_factor ((int, float): The coefficient of backbone's loss, which is computed by loss function. Default: 1.
        bnn_factor (int, float): The coefficient of KL loss, which is KL divergence of Bayesian layer. Default: 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        ...         self.bn = nn.BatchNorm2d(64)
        ...         self.relu = nn.ReLU()
        ...         self.flatten = nn.Flatten()
        ...         self.fc = nn.Dense(64*224*224, 12) # padding=0
        ...
        ...     def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.bn(x)
        ...         x = self.relu(x)
        ...         x = self.flatten(x)
        ...         out = self.fc(x)
        ...         return out
        >>>
        >>> net = Net()
        >>> criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, criterion)
        >>> train_network = TrainOneStepCell(net_with_loss, optim)
        >>> bnn_transformer = TransformToBNN(train_network, 60000, 0.0001)
    """

    def __init__(self, trainable_dnn, dnn_factor=1, bnn_factor=1):
        if isinstance(dnn_factor, bool) or not isinstance(dnn_factor, (int, float)):
            raise TypeError('The type of `dnn_factor` should be `int` or `float`')
        if dnn_factor < 0:
            raise ValueError('The value of `dnn_factor` should >= 0')

        if isinstance(bnn_factor, bool) or not isinstance(bnn_factor, (int, float)):
            raise TypeError('The type of `bnn_factor` should be `int` or `float`')
        if bnn_factor < 0:
            raise ValueError('The value of `bnn_factor` should >= 0')

        net_with_loss = trainable_dnn.network
        self.optimizer = trainable_dnn.optimizer
        self.backbone = net_with_loss.backbone_network
        self.loss_fn = getattr(net_with_loss, "_loss_fn")
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor

    def transform_to_bnn_model(self,
                               get_dense_args=lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias,
                                                          "out_channels": dp.out_channels, "activation": dp.activation},
                               get_conv_args=lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels,
                                                         "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size,
                                                         "stride": dp.stride, "has_bias": dp.has_bias,
                                                         "padding": dp.padding, "dilation": dp.dilation,
                                                         "group": dp.group},
                               add_dense_args=None,
                               add_conv_args=None):
        r"""
        Transform the whole DNN model to BNN model, and wrap BNN model by TrainOneStepCell.

        Args:
            get_dense_args: The arguments gotten from the DNN full connection layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "has_bias": dp.has_bias}.
            get_conv_args: The arguments gotten from the DNN convolutional layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode,
                "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias}.
            add_dense_args (dict): The new arguments added to BNN full connection layer. Note that the arguments in
                `add_dense_args` must not duplicate arguments in `get_dense_args`. Default: None.
            add_conv_args (dict): The new arguments added to BNN convolutional layer. Note that the arguments in
                `add_conv_args` must not duplicate arguments in `get_conv_args`. Default: None.

        Returns:
            Cell, a trainable BNN model wrapped by TrainOneStepCell.

        Supported Platforms:
        ``Ascend`` ``GPU``

        Examples:
            >>> net = Net()
            >>> criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> net_with_loss = WithLossCell(net, criterion)
            >>> train_network = TrainOneStepCell(net_with_loss, optim)
            >>> bnn_transformer = TransformToBNN(train_network, 60000, 0.1)
            >>> train_bnn_network = bnn_transformer.transform_to_bnn_model()
        """
        if not add_dense_args:
            add_dense_args = {}
        if not add_conv_args:
            add_conv_args = {}

        self._replace_all_bnn_layers(self.backbone, get_dense_args, get_conv_args, add_dense_args, add_conv_args)

        # rename layers of BNN model to prevent duplication of names
        for value, param in self.backbone.parameters_and_names():
            param.name = value

        bnn_with_loss = WithBNNLossCell(self.backbone, self.loss_fn, self.dnn_factor, self.bnn_factor)
        bnn_optimizer = self._create_optimizer_with_bnn_params()
        train_bnn_network = TrainOneStepCell(bnn_with_loss, bnn_optimizer)
        return train_bnn_network

    def transform_to_bnn_layer(self, dnn_layer_type, bnn_layer_type, get_args=None, add_args=None):
        r"""
        Transform a specific type of layers in DNN model to corresponding BNN layer.

        Args:
            dnn_layer_type (Cell): The type of DNN layer to be transformed to BNN layer. The optional values are
                nn.Dense and nn.Conv2d.
            bnn_layer_type (Cell): The type of BNN layer to be transformed to. The optional values are
                DenseReparam and ConvReparam.
            get_args: The arguments gotten from the DNN layer. Default: None.
            add_args (dict): The new arguments added to BNN layer. Note that the arguments in `add_args` must not
                duplicate arguments in `get_args`. Default: None.

        Returns:
            Cell, a trainable model wrapped by TrainOneStepCell, whose specific type of layer is transformed to the
            corresponding bayesian layer.

        Supported Platforms:
        ``Ascend`` ``GPU``

        Examples:
            >>> net = Net()
            >>> criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> net_with_loss = WithLossCell(net, criterion)
            >>> train_network = TrainOneStepCell(net_with_loss, optim)
            >>> bnn_transformer = TransformToBNN(train_network, 60000, 0.1)
            >>> train_bnn_network = bnn_transformer.transform_to_bnn_layer(Dense, DenseReparam)
        """
        if dnn_layer_type.__name__ not in ["Dense", "Conv2d"]:
            raise ValueError(' \'dnn_layer\'' + str(dnn_layer_type) +
                             ', should be one of values in \'nn.Dense\', \'nn.Conv2d\'.')

        if bnn_layer_type.__name__ not in ["DenseReparam", "ConvReparam"]:
            raise ValueError(' \'bnn_layer\'' + str(bnn_layer_type) +
                             ', should be one of values in \'DenseReparam\', \'ConvReparam\'.')

        dnn_layer_type = getattr(layer, dnn_layer_type.__name__)
        bnn_layer_type = getattr(bnn_layers, bnn_layer_type.__name__)

        if not get_args:
            if dnn_layer_type.__name__ == "Dense":
                get_args = self._get_dense_args
            else:
                get_args = self._get_conv_args

        if not add_args:
            add_args = {}

        self._replace_specified_dnn_layers(self.backbone, dnn_layer_type, bnn_layer_type, get_args, add_args)
        for value, param in self.backbone.parameters_and_names():
            param.name = value

        bnn_with_loss = WithBNNLossCell(self.backbone, self.loss_fn, self.dnn_factor, self.bnn_factor)
        bnn_optimizer = self._create_optimizer_with_bnn_params()

        train_bnn_network = TrainOneStepCell(bnn_with_loss, bnn_optimizer)
        return train_bnn_network

    def _get_dense_args(self, dense_layer):
        """Get arguments from dense layer."""
        dense_args = {"in_channels": dense_layer.in_channels, "has_bias": dense_layer.has_bias,
                      "out_channels": dense_layer.out_channels, "activation": dense_layer.activation}
        return dense_args

    def _get_conv_args(self, conv_layer):
        """Get arguments from conv2d layer."""
        conv_args = {"in_channels": conv_layer.in_channels, "out_channels": conv_layer.out_channels,
                     "pad_mode": conv_layer.pad_mode, "kernel_size": conv_layer.kernel_size,
                     "stride": conv_layer.stride, "has_bias": conv_layer.has_bias,
                     "padding": conv_layer.padding, "dilation": conv_layer.dilation,
                     "group": conv_layer.group}
        return conv_args

    def _create_optimizer_with_bnn_params(self):
        """Create new optimizer that contains bnn trainable parameters."""
        name = self.optimizer.__class__.__name__
        modules = optim.__all__

        if name not in modules:
            raise TypeError('The optimizer can be {}, but got {}'.format(str(modules), name))

        optimizer = getattr(optim, name)

        args = {'params': self.backbone.trainable_params()}
        params = optimizer.__init__.__code__.co_varnames
        _params = self.optimizer.__dict__['_params']
        for param in params:
            if param in _params:
                args[param] = self.optimizer.__getattr__(param).data.asnumpy().tolist()

        new_optimizer = optimizer(**args)
        return new_optimizer

    def _replace_all_bnn_layers(self, backbone, get_dense_args, get_conv_args, add_dense_args, add_conv_args):
        """Replace both dense layer and conv2d layer in DNN model to bayesian layers."""
        for name, cell in backbone.name_cells().items():
            if isinstance(cell, nn.Dense):
                dense_args = get_dense_args(cell)
                new_layer = DenseReparam(**dense_args, **add_dense_args)
                setattr(backbone, name, new_layer)
            elif isinstance(cell, nn.Conv2d):
                conv_args = get_conv_args(cell)
                new_layer = ConvReparam(**conv_args, **add_conv_args)
                setattr(backbone, name, new_layer)
            else:
                self._replace_all_bnn_layers(cell, get_dense_args, get_conv_args, add_dense_args,
                                             add_conv_args)

    def _replace_specified_dnn_layers(self, backbone, dnn_layer, bnn_layer, get_args, add_args):
        """Convert a specific type of layers in DNN model to corresponding bayesian layers."""
        for name, cell in backbone.name_cells().items():
            if isinstance(cell, dnn_layer):
                args = get_args(cell)
                new_layer = bnn_layer(**args, **add_args)
                setattr(backbone, name, new_layer)
            else:
                self._replace_specified_dnn_layers(cell, dnn_layer, bnn_layer, get_args, add_args)
