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
"""
Quantization aware training

User can use quantization aware to train a model. MindSpore supports quantization aware training,
which models quantization errors in both the forward and backward passes using fake-quantization
operations. Note that the entire computation is carried out in floating point. At the end of quantization
aware training, MindSpore provides conversion functions to convert the trained model into lower precision.
"""

import re

import mindspore.context as context

from ... import nn, ops
from ..._checkparam import Validator, Rel
from ...nn.layer import quant
from ...ops import functional as F
from ..common import QuantDtype
from .quantizer import Quantizer, OptimizeOption


__all__ = ["QuantizationAwareTraining", "create_quant_config"]


def create_quant_config(quant_observer=(nn.FakeQuantWithMinMaxObserver, nn.FakeQuantWithMinMaxObserver),
                        quant_delay=(0, 0),
                        quant_dtype=(QuantDtype.INT8, QuantDtype.INT8),
                        per_channel=(False, False),
                        symmetric=(False, False),
                        narrow_range=(False, False)):
    r"""
    Config the observer type of weights and data flow with quant params.

    Args:
        quant_observer (Union[Observer, list, tuple]): The observer type to do quantization. The first element
            represents weights and second element represents data flow.
            Default: (nn.FakeQuantWithMinMaxObserver, nn.FakeQuantWithMinMaxObserver)
        quant_delay (Union[int, list, tuple]): Number of steps after which weights and activations are quantized during
            eval. The first element represents weights and second element represents data flow. Default: (0, 0)
        quant_dtype (Union[QuantDtype, list, tuple]): Datatype to use for quantize weights and activations. The first
            element represents weights and second element represents data flow.
            Default: (QuantDtype.INT8, QuantDtype.INT8)
        per_channel (Union[bool, list, tuple]):  Quantization granularity based on layer or on channel. If `True`
            then base on per channel otherwise base on per layer. The first element represents weights
            and second element represents data flow. Default: (False, False)
        symmetric (Union[bool, list, tuple]): Whether the quantization algorithm is symmetric or not. If `True` then
            base on symmetric otherwise base on asymmetric. The first element represents weights and second
            element represents data flow. Default: (False, False)
        narrow_range (Union[bool, list, tuple]): Whether the quantization algorithm uses narrow range or not.
            The first element represents weights and the second element represents data flow. Default: (False, False)

    Returns:
        QuantConfig, Contains the observer type of weight and activation.
    """
    weight_observer = quant_observer[0].partial_init(quant_delay=quant_delay[0], quant_dtype=quant_dtype[0],
                                                     per_channel=per_channel[0], symmetric=symmetric[0],
                                                     narrow_range=narrow_range[0])
    act_observer = quant_observer[-1].partial_init(quant_delay=quant_delay[-1], quant_dtype=quant_dtype[-1],
                                                   per_channel=per_channel[-1], symmetric=symmetric[-1],
                                                   narrow_range=narrow_range[-1])
    return quant.QuantConfig(weight=weight_observer, activation=act_observer)


class _AddFakeQuantInput(nn.Cell):
    """
    Add FakeQuant OP at input of the network. Only support one input case.
    """

    def __init__(self, network, quant_delay=0):
        super(_AddFakeQuantInput, self).__init__(auto_prefix=False)
        self.fake_quant_input = quant.FakeQuantWithMinMaxObserver(min_init=-6, max_init=6,
                                                                  quant_delay=quant_delay, ema=True)
        self.fake_quant_input.update_parameters_name('fake_quant_input.')
        self.network = network

    def construct(self, data):
        data = self.fake_quant_input(data)
        output = self.network(data)
        return output


class _AddFakeQuantAfterSubCell(nn.Cell):
    """
    Add FakeQuant OP after of the sub Cell.
    """

    def __init__(self, subcell, **kwargs):
        super(_AddFakeQuantAfterSubCell, self).__init__(auto_prefix=False)
        self.subcell = subcell
        self.fake_quant_act = quant.FakeQuantWithMinMaxObserver(min_init=-6,
                                                                max_init=6,
                                                                ema=True,
                                                                quant_dtype=kwargs["quant_dtype"],
                                                                quant_delay=kwargs["quant_delay"],
                                                                per_channel=kwargs["per_channel"],
                                                                symmetric=kwargs["symmetric"],
                                                                narrow_range=kwargs["narrow_range"])

    def construct(self, *data):
        output = self.subcell(*data)
        output = self.fake_quant_act(output)
        return output


class QuantizationAwareTraining(Quantizer):
    r"""
    Quantizer for quantization aware training.

    Args:
        bn_fold (bool): Flag to used bn fold ops for simulation inference operation. Default: True.
        freeze_bn (int): Number of steps after which BatchNorm OP parameters used total mean and variance. Default: 1e7.
        quant_delay (Union[int, list, tuple]): Number of steps after which weights and activations are quantized during
            eval. The first element represents weights and second element represents data flow. Default: (0, 0)
        quant_dtype (Union[QuantDtype, list, tuple]): Datatype to use for quantize weights and activations. The first
            element represents weights and second element represents data flow.
            Default: (QuantDtype.INT8, QuantDtype.INT8)
        per_channel (Union[bool, list, tuple]):  Quantization granularity based on layer or on channel. If `True`
            then base on per channel otherwise base on per layer. The first element represents weights
            and second element represents data flow. Default: (False, False)
        symmetric (Union[bool, list, tuple]): Whether the quantization algorithm is symmetric or not. If `True` then
            base on symmetric otherwise base on asymmetric. The first element represents weights and second
            element represents data flow. Default: (False, False)
        narrow_range (Union[bool, list, tuple]): Whether the quantization algorithm uses narrow range or not.
            The first element represents weights and the second element represents data flow. Default: (False, False)
        optimize_option (Union[OptimizeOption, list, tuple]): Specifies the quant algorithm and options, currently only
            support QAT. Default: OptimizeOption.QAT
        one_conv_fold (bool): Flag to used one conv bn fold ops for simulation inference operation. Default: True.

    Examples:
        >>> class LeNet5(nn.Cell):
        ...     def __init__(self, num_class=10, channel=1):
        ...         super(LeNet5, self).__init__()
        ...         self.type = "fusion"
        ...         self.num_class = num_class
        ...
        ...         # change `nn.Conv2d` to `nn.Conv2dBnAct`
        ...         self.conv1 = nn.Conv2dBnAct(channel, 6, 5, pad_mode='valid', activation='relu')
        ...         self.conv2 = nn.Conv2dBnAct(6, 16, 5, pad_mode='valid', activation='relu')
        ...         # change `nn.Dense` to `nn.DenseBnAct`
        ...         self.fc1 = nn.DenseBnAct(16 * 5 * 5, 120, activation='relu')
        ...         self.fc2 = nn.DenseBnAct(120, 84, activation='relu')
        ...         self.fc3 = nn.DenseBnAct(84, self.num_class)
        ...
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.flatten = nn.Flatten()
        ...
        ...     def construct(self, x):
        ...         x = self.conv1(x)
        ...         x = self.max_pool2d(x)
        ...         x = self.conv2(x)
        ...         x = self.max_pool2d(x)
        ...         x = self.flatten(x)
        ...         x = self.fc1(x)
        ...         x = self.fc2(x)
        ...         x = self.fc3(x)
        ...         return x
        ...
        >>> net = LeNet5()
        >>> quantizer = QuantizationAwareTraining(bn_fold=False, per_channel=[True, False], symmetric=[True, False])
        >>> net_qat = quantizer.quantize(net)
    """
    __quant_op_name__ = ["Add", "Sub", "Mul", "RealDiv"]

    def __init__(self,
                 bn_fold=True,
                 freeze_bn=10000000,
                 quant_delay=(0, 0),
                 quant_dtype=(QuantDtype.INT8, QuantDtype.INT8),
                 per_channel=(False, False),
                 symmetric=(False, False),
                 narrow_range=(False, False),
                 optimize_option=OptimizeOption.QAT,
                 one_conv_fold=True):
        """Init for QuantizationAwareTraining quantizer"""
        super(QuantizationAwareTraining, self).__init__(optimize_option=optimize_option)

        def convert2list(name, value):
            if not isinstance(value, list) and not isinstance(value, tuple):
                value = [value]
            elif len(value) > 2:
                raise ValueError("input `{}` len should less then 2".format(name))
            return value

        quant_delay = convert2list("quant delay", quant_delay)
        quant_dtype = convert2list("quant dtype", quant_dtype)
        per_channel = convert2list("per channel", per_channel)
        symmetric = convert2list("symmetric", symmetric)
        narrow_range = convert2list("narrow range", narrow_range)

        self.weight_qdelay = Validator.check_non_negative_int(quant_delay[0], "quant delay")
        self.act_qdelay = Validator.check_int(quant_delay[-1], 0, Rel.GE, "quant delay")
        self.bn_fold = Validator.check_bool(bn_fold, "bn fold")
        self.freeze_bn = Validator.check_non_negative_int(freeze_bn, "freeze bn")
        self.weight_dtype = Validator.check_isinstance("weights dtype", quant_dtype[0], QuantDtype)
        self.act_dtype = Validator.check_isinstance("activations dtype", quant_dtype[-1], QuantDtype)
        self.weight_channel = Validator.check_bool(per_channel[0], "per channel")
        self.act_channel = Validator.check_bool(per_channel[-1], "per channel")
        self.weight_symmetric = Validator.check_bool(symmetric[0], "symmetric")
        self.act_symmetric = Validator.check_bool(symmetric[-1], "symmetric")
        self.weight_range = Validator.check_bool(narrow_range[0], "narrow range")
        self.act_range = Validator.check_bool(narrow_range[-1], "narrow range")
        self.one_conv_fold = Validator.check_bool(one_conv_fold, "one conv fold")
        self._convert_method_map = {nn.Conv2dBnAct: self._convert_conv,
                                    nn.DenseBnAct: self._convert_dense}
        self.quant_config = create_quant_config(quant_delay=quant_delay,
                                                quant_dtype=quant_dtype,
                                                per_channel=per_channel,
                                                symmetric=symmetric,
                                                narrow_range=narrow_range)

    def _convert_op_name(self, name):
        pattern = re.compile(r'([A-Z]{1})')
        name_new = re.sub(pattern, r'_\1', name).lower()
        if name_new[0] == '_':
            name_new = name_new[1:]
        return name_new

    def quantize(self, network):
        """
        Quant API to convert input network to a quantization aware training network

        Args:
            network (Cell): network to be quantized.

        Examples:
            >>> net = Net()
            >>> quantizer = QuantizationAwareTraining()
            >>> net_qat = quantizer.quantize(net)
        """
        support_device = ["Ascend", "GPU"]
        if context.get_context('device_target') not in support_device:
            raise KeyError("Unsupported {} device target.".format(context.get_context('device_target')))

        if OptimizeOption.QAT in self.optimize_option:
            network.update_cell_prefix()
            network = self._convert_subcells2quant(network)
            network.update_cell_type("quant")
        return network

    def _convert_subcells2quant(self, network):
        """
        convert sub cell like `Conv2dBnAct` and `DenseBnAct` to quant cell
        """
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            elif isinstance(subcell, (nn.Conv2dBnAct, nn.DenseBnAct)):
                prefix = subcell.param_prefix
                new_subcell = self._convert_method_map[type(subcell)](subcell)
                new_subcell.update_parameters_name(prefix + '.')
                network.insert_child_to_cell(name, new_subcell)
                change = True
            else:
                self._convert_subcells2quant(subcell)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())

        # add FakeQuant OP after OP in while list
        add_list = []
        for name in network.__dict__:
            if name[0] == '_':
                continue
            attr = network.__dict__[name]
            if isinstance(attr, ops.Primitive) and attr.name in self.__quant_op_name__:
                add_list.append((name, attr))
        for name, prim_op in add_list:
            prefix = name
            add_quant = _AddFakeQuantAfterSubCell(prim_op,
                                                  quant_dtype=self.act_dtype,
                                                  quant_delay=self.act_qdelay,
                                                  per_channel=self.act_channel,
                                                  symmetric=self.act_symmetric,
                                                  narrow_range=self.act_range)
            prefix = self._convert_op_name(prim_op.name)
            if network.param_prefix:
                prefix = '.'.join([network.param_prefix, self._convert_op_name(prim_op.name)])
            add_quant.update_parameters_name(prefix + '.')
            del network.__dict__[name]
            network.insert_child_to_cell(name, add_quant)
        return network

    def _convert_conv(self, subcell):
        """
        convert Conv2d cell to quant cell
        """
        conv_inner = subcell.conv
        if subcell.has_bn:
            if self.bn_fold:
                bn_inner = subcell.batchnorm
                if self.one_conv_fold:
                    conv_inner = quant.Conv2dBnFoldQuantOneConv(conv_inner.in_channels,
                                                                conv_inner.out_channels,
                                                                kernel_size=conv_inner.kernel_size,
                                                                stride=conv_inner.stride,
                                                                pad_mode=conv_inner.pad_mode,
                                                                padding=conv_inner.padding,
                                                                dilation=conv_inner.dilation,
                                                                group=conv_inner.group,
                                                                eps=bn_inner.eps,
                                                                momentum=bn_inner.momentum,
                                                                has_bias=conv_inner.has_bias,
                                                                bias_init=conv_inner.bias_init,
                                                                quant_config=self.quant_config,
                                                                quant_dtype=self.weight_dtype,
                                                                fake=True)
                else:
                    conv_inner = quant.Conv2dBnFoldQuant(conv_inner.in_channels,
                                                         conv_inner.out_channels,
                                                         kernel_size=conv_inner.kernel_size,
                                                         stride=conv_inner.stride,
                                                         pad_mode=conv_inner.pad_mode,
                                                         padding=conv_inner.padding,
                                                         dilation=conv_inner.dilation,
                                                         group=conv_inner.group,
                                                         eps=bn_inner.eps,
                                                         momentum=bn_inner.momentum,
                                                         has_bias=conv_inner.has_bias,
                                                         bias_init=conv_inner.bias_init,
                                                         freeze_bn=self.freeze_bn,
                                                         quant_config=self.quant_config,
                                                         quant_dtype=self.weight_dtype,
                                                         fake=True)
                # change original network BatchNormal OP parameters to quant network
                conv_inner.gamma = subcell.batchnorm.gamma
                conv_inner.beta = subcell.batchnorm.beta
                conv_inner.moving_mean = subcell.batchnorm.moving_mean
                conv_inner.moving_variance = subcell.batchnorm.moving_variance
                del subcell.batchnorm
                subcell.batchnorm = None
                subcell.has_bn = False
            else:
                bn_inner = subcell.batchnorm
                conv_inner = quant.Conv2dBnWithoutFoldQuant(conv_inner.in_channels,
                                                            conv_inner.out_channels,
                                                            kernel_size=conv_inner.kernel_size,
                                                            stride=conv_inner.stride,
                                                            pad_mode=conv_inner.pad_mode,
                                                            padding=conv_inner.padding,
                                                            dilation=conv_inner.dilation,
                                                            group=conv_inner.group,
                                                            eps=bn_inner.eps,
                                                            momentum=bn_inner.momentum,
                                                            has_bias=conv_inner.has_bias,
                                                            bias_init=conv_inner.bias_init,
                                                            quant_config=self.quant_config,
                                                            quant_dtype=self.weight_dtype)
                # change original network BatchNormal OP parameters to quant network
                conv_inner.batchnorm.gamma = subcell.batchnorm.gamma
                conv_inner.batchnorm.beta = subcell.batchnorm.beta
                conv_inner.batchnorm.moving_mean = subcell.batchnorm.moving_mean
                conv_inner.batchnorm.moving_variance = subcell.batchnorm.moving_variance
                del subcell.batchnorm
                subcell.batchnorm = None
                subcell.has_bn = False
        else:
            conv_inner = quant.Conv2dQuant(conv_inner.in_channels,
                                           conv_inner.out_channels,
                                           kernel_size=conv_inner.kernel_size,
                                           stride=conv_inner.stride,
                                           pad_mode=conv_inner.pad_mode,
                                           padding=conv_inner.padding,
                                           dilation=conv_inner.dilation,
                                           group=conv_inner.group,
                                           has_bias=conv_inner.has_bias,
                                           quant_config=self.quant_config,
                                           quant_dtype=self.weight_dtype)
        # change original network Conv2D OP parameters to quant network
        conv_inner.weight = subcell.conv.weight
        if subcell.conv.has_bias:
            conv_inner.bias = subcell.conv.bias
        subcell.conv = conv_inner
        if subcell.has_act and subcell.activation is not None:
            subcell.activation = self._convert_activation(subcell.activation)
        elif subcell.after_fake:
            subcell.has_act = True
            subcell.activation = _AddFakeQuantAfterSubCell(F.identity,
                                                           quant_dtype=self.act_dtype,
                                                           quant_delay=self.act_qdelay,
                                                           per_channel=self.act_channel,
                                                           symmetric=self.act_symmetric,
                                                           narrow_range=self.act_range)
        return subcell

    def _convert_dense(self, subcell):
        """
        convert dense cell to quant cell
        """
        dense_inner = subcell.dense
        dense_inner = quant.DenseQuant(dense_inner.in_channels,
                                       dense_inner.out_channels,
                                       has_bias=dense_inner.has_bias,
                                       quant_config=self.quant_config,
                                       quant_dtype=self.weight_dtype)
        # change original network Dense OP parameters to quant network
        dense_inner.weight = subcell.dense.weight
        if subcell.dense.has_bias:
            dense_inner.bias = subcell.dense.bias
        subcell.dense = dense_inner
        if subcell.has_act and subcell.activation is not None:
            subcell.activation = self._convert_activation(subcell.activation)
        elif subcell.after_fake:
            subcell.has_act = True
            subcell.activation = _AddFakeQuantAfterSubCell(F.identity,
                                                           quant_dtype=self.act_dtype,
                                                           quant_delay=self.act_qdelay,
                                                           per_channel=self.act_channel,
                                                           symmetric=self.act_symmetric,
                                                           narrow_range=self.act_range)
        return subcell

    def _convert_activation(self, activation):
        """
        convert activation cell to quant cell
        """
        act_class = activation.__class__
        act_list = [nn.ReLU, nn.ReLU6, nn.Sigmoid]
        act_list_with_fake_before = [nn.LeakyReLU, nn.HSigmoid, nn.HSwish]
        if act_class in act_list:
            return quant.ActQuant(activation=activation,
                                  quant_config=self.quant_config,
                                  quant_dtype=self.act_dtype)
        if act_class in act_list_with_fake_before:
            return quant.ActQuant(activation=activation,
                                  ema=True,
                                  fake_before=True,
                                  quant_config=self.quant_config,
                                  quant_dtype=self.act_dtype)
        raise ValueError("Unsupported activation in auto quant: ", act_class)
