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
import numpy as np
from ... import nn, ops
from ..._checkparam import Validator, Rel
from ...nn.layer import quant
from ...ops import functional as F
from ..common import QuantDtype
from .quantizer import Quantizer, OptimizeOption
from .quant_utils import compute_kl_threshold


__all__ = ["QuantizationAwareTraining", "create_quant_config"]


def create_quant_config(quant_observer=(nn.FakeQuantWithMinMaxObserver, nn.FakeQuantWithMinMaxObserver),
                        quant_delay=(0, 0),
                        quant_dtype=(QuantDtype.INT8, QuantDtype.INT8),
                        per_channel=(False, False),
                        symmetric=(False, False),
                        narrow_range=(False, False),
                        mode="DEFAULT"):
    r"""
    Config the observer type of weights and data flow with quant parameters.

    Args:
        quant_observer (Union[Observer, list, tuple]): The types of observer for quantization. The first element
            applies to weights and the second applies to data flow. Currently, only
            :class:`FakeQuantWithMinMaxObserver` supported.
            Default: (nn.FakeQuantWithMinMaxObserver, nn.FakeQuantWithMinMaxObserver).
        quant_delay (Union[int, list, tuple]): Number of steps after which weights and activations are quantized
            during train and eval. The first element represents weights and the second element represents data flow.
            Default: (0, 0).
        quant_dtype (Union[QuantDtype, list, tuple]): Datatype used to quantize weights and activations. The first
            element represents weights and the second element represents data flow.
            Default: (QuantDtype.INT8, QuantDtype.INT8).
        per_channel (Union[bool, list, tuple]):  Quantization granularity based on layer or on channel. If `True`
            then base on per channel, otherwise base on per layer. The first element represents weights
            and the second element represents data flow, and the second element must be `False` now.
            Default: (False, False).
        symmetric (Union[bool, list, tuple]): Whether the quantization algorithm is symmetric or not. If `True` then
            base on symmetric, otherwise base on asymmetric. The first element represents weights and the second
            element represents data flow. Default: (False, False).
        narrow_range (Union[bool, list, tuple]): Whether the quantization algorithm uses narrow range or not.
            The first element represents weights and the second element represents data flow.
            Default: (False, False).
        mode (str): Optional quantization mode, currently only `DEFAULT`(QAT) and `LEARNED_SCALE` are supported.
            Default: ("DEFAULT").

    Returns:
        QuantConfig, contains the observer type of weight and activation.

    Raises:
        ValueError: If the second element of `per_channel` is not `False`.
    """
    if per_channel[-1]:
        raise ValueError("Arg 'per_channel' second element must be 'False'.")
    weight_observer = quant_observer[0].partial_init(quant_delay=quant_delay[0], quant_dtype=quant_dtype[0],
                                                     per_channel=per_channel[0], symmetric=symmetric[0],
                                                     narrow_range=narrow_range[0], mode=mode)
    act_observer = quant_observer[-1].partial_init(quant_delay=quant_delay[-1], quant_dtype=quant_dtype[-1],
                                                   per_channel=per_channel[-1], symmetric=symmetric[-1],
                                                   narrow_range=narrow_range[-1], mode=mode)
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
        self.mode = "DEFAULT"
        self.max_init = 6
        self.min_init = -6

        if OptimizeOption.LEARNED_SCALE in kwargs["optimize_option"]:
            self.mode = "LEARNED_SCALE"
            self.max_init = 16
            self.min_init = -16

        self.fake_quant_act = quant.FakeQuantWithMinMaxObserver(min_init=self.min_init,
                                                                max_init=self.max_init,
                                                                ema=True,
                                                                quant_dtype=kwargs["quant_dtype"],
                                                                quant_delay=kwargs["quant_delay"],
                                                                per_channel=kwargs["per_channel"],
                                                                symmetric=kwargs["symmetric"],
                                                                narrow_range=kwargs["narrow_range"],
                                                                mode=self.mode)

    def construct(self, *data):
        output = self.subcell(*data)
        output = self.fake_quant_act(output)
        return output


class QuantizationAwareTraining(Quantizer):
    r"""
    Quantizer for quantization aware training.

    Args:
        bn_fold (bool): Whether to use bn fold ops for simulation inference operation. Default: True.
        freeze_bn (int): Number of steps after which BatchNorm OP parameters fixed to global mean and variance.
            Default: 1e7.
        quant_delay (Union[int, list, tuple]): Number of steps after which weights and activations are quantized
            during train and eval. The first element represents weights and the second element represents data flow.
            Default: (0, 0).
        quant_dtype (Union[QuantDtype, list, tuple]): Datatype used to quantize weights and activations. The first
            element represents weights and the second element represents data flow. It is necessary to consider the
            precision support of hardware devices in the practical quantization infer scenario.
            Default: (QuantDtype.INT8, QuantDtype.INT8).
        per_channel (Union[bool, list, tuple]):  Quantization granularity based on layer or on channel. If `True`
            then base on per channel, otherwise base on per layer. The first element represents weights and the
            second element represents data flow, and the second element must be `False` now. Default: (False, False).
        symmetric (Union[bool, list, tuple]): Whether the quantization algorithm is symmetric or not. If `True` then
            base on symmetric, otherwise base on asymmetric. The first element represents weights and the second
            element represents data flow. Default: (False, False).
        narrow_range (Union[bool, list, tuple]): Whether the quantization algorithm uses narrow range or not.
            The first element represents weights and the second element represents data flow.
            Default: (False, False).
        optimize_option (Union[OptimizeOption, list, tuple]): Specifies the quant algorithm and options, currently
            only support `QAT` and `LEARNED_SCALE` (Note that, if both `QAT` and `LEARNED_SCALE` are configured,
            `LEARNED_SCALE` has a higher priority. `LEARNED_SCALE` currently only work under some constraints, which
            includes: freeze_bn=0, quant_delay=0, symmetric=True, narrow_range=True, More specifically, for operators
            such as Relu and Relu6, which only have positive values, we add a negative truncation to optimize this
            scenario, and narrow_range will automatically match to False). Default: OptimizeOption.QAT.
        one_conv_fold (bool): Whether to use one conv bn fold ops for simulation inference operation. Default: True.

    Raises:
        TypeError: If the element of `quant_delay` or `freeze_bn` is not int.
        TypeError: If `bn_fold`, `one_conv_fold` or the element of `per_channel`, `symmetric`, `narrow_range`
            is not bool.
        TypeError: If the element of `quant_dtype` is not `QuantDtype`.
        ValueError: If the length of `quant_delay`, `quant_dtype`, `per_channel`, `symmetric` or `narrow_range` is
            not less than 2.
        ValueError: If the `optimize_option` is `LEARNED_SCALE` and `freeze_bn` is not equal to 0.
        ValueError: If the `optimize_option` is `LEARNED_SCALE` and `symmetric` is not (True, True).
        ValueError: If the `optimize_option` is `LEARNED_SCALE` and `narrow_range` is not (True, True).
        ValueError: If the `optimize_option` is `LEARNED_SCALE` and `quant_delay` is not (0, 0).

    Examples:
        >>> from mindspore.compression.quant import QuantizationAwareTraining
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
    __quant_op_name__ = ["Add", "Sub", "Mul", "RealDiv", "ReduceMean"]

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
        self.mode = "DEFAULT"
        if OptimizeOption.LEARNED_SCALE in self.optimize_option:
            self.mode = "LEARNED_SCALE"
            if not self.weight_symmetric or not self.act_symmetric:
                raise ValueError("OptimizeOption.LEARNED_SCALE currently only support "
                                 "symmetric=(True, True) for quant")
            if not self.weight_range or not self.act_range:
                raise ValueError("OptimizeOption.LEARNED_SCALE currently only support narrow_range=(True, True) "
                                 "for quant")
            if self.freeze_bn != 0:
                raise ValueError("OptimizeOption.LEARNED_SCALE currently only support freeze_bn equal to 0, "
                                 "but get freeze_bn={}".format(self.freeze_bn))
            if self.weight_qdelay != 0 or self.act_qdelay != 0:
                raise ValueError("OptimizeOption.LEARNED_SCALE currently only support quant_delay=(0, 0)")
        self.quant_config = create_quant_config(quant_delay=quant_delay,
                                                quant_dtype=quant_dtype,
                                                per_channel=per_channel,
                                                symmetric=symmetric,
                                                narrow_range=narrow_range,
                                                mode=self.mode)
        self.eps = 1e-5

    @staticmethod
    def _convert_op_name(name):
        pattern = re.compile(r'([A-Z]{1})')
        name_new = re.sub(pattern, r'_\1', name).lower()
        if name_new[0] == '_':
            name_new = name_new[1:]
        return name_new

    def quantize(self, network):
        """
        Quant API to convert input network to a quantization aware training network.

        Note:
            Please refer to the Examples of class: `mindspore.compression.quant.QuantizationAwareTraining`.

        Args:
            network (Cell): network to be quantized.

        Returns:
            Cell, a quantization aware training network.

        Raises:
            KeyError: If the `device_target` set in context is not in `support_device`.
        """
        support_device = ["Ascend", "GPU"]
        if context.get_context('device_target') not in support_device:
            raise KeyError("Unsupported {} device target.".format(context.get_context('device_target')))

        if OptimizeOption.QAT in self.optimize_option or OptimizeOption.LEARNED_SCALE in self.optimize_option:
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

        # add FakeQuant OP after OP in white list, but not including those wrapped in the below quantization cell.
        if isinstance(network, (nn.FakeQuantWithMinMaxObserver,
                                nn.Conv2dBnFoldQuantOneConv,
                                nn.Conv2dBnFoldQuant,
                                nn.Conv2dBnWithoutFoldQuant,
                                nn.Conv2dQuant,
                                nn.DenseQuant,
                                nn.ActQuant,
                                nn.TensorAddQuant,
                                nn.MulQuant)):
            return network

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
                                                  narrow_range=self.act_range,
                                                  optimize_option=self.optimize_option)
            if network.param_prefix:
                prefix = '.'.join([network.param_prefix, prefix])
            add_quant.update_parameters_name(prefix + '.')
            del network.__dict__[name]
            network.insert_child_to_cell(name, add_quant)
        return network

    def _convert_conv(self, subcell):
        """
        convert Conv2d cell to quant cell
        """
        min_init = -6
        max_init = 6
        if OptimizeOption.LEARNED_SCALE in self.optimize_option:
            subcell_weight_para = subcell.conv.weight.data.asnumpy()
            if subcell.has_bn:
                scale_factor = (subcell.batchnorm.gamma.data.asnumpy() /
                                np.sqrt(subcell.batchnorm.moving_variance.data.asnumpy() + self.eps))
                subcell_weight_para = subcell_weight_para * scale_factor.reshape(-1, 1, 1, 1)
            min_init, max_init = self._kl_init(subcell_weight_para, self.weight_dtype)
        self.quant_config = self.quant_config._replace(
            weight=self.quant_config.weight.partial_init(min_init=min_init, max_init=max_init))

        conv_inner = subcell.conv
        if subcell.has_bn:
            bn_inner = subcell.batchnorm
            if self.bn_fold:
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
                                                                momentum=1 - bn_inner.momentum,
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
                                                         momentum=1 - bn_inner.momentum,
                                                         has_bias=conv_inner.has_bias,
                                                         bias_init=conv_inner.bias_init,
                                                         freeze_bn=self.freeze_bn,
                                                         quant_config=self.quant_config,
                                                         quant_dtype=self.weight_dtype,
                                                         fake=True)
                # change original network Batch Normalization OP parameters to quant network
                conv_inner.gamma = subcell.batchnorm.gamma
                conv_inner.beta = subcell.batchnorm.beta
                conv_inner.moving_mean = subcell.batchnorm.moving_mean
                conv_inner.moving_variance = subcell.batchnorm.moving_variance
            else:
                conv_inner = quant.Conv2dBnWithoutFoldQuant(conv_inner.in_channels,
                                                            conv_inner.out_channels,
                                                            kernel_size=conv_inner.kernel_size,
                                                            stride=conv_inner.stride,
                                                            pad_mode=conv_inner.pad_mode,
                                                            padding=conv_inner.padding,
                                                            dilation=conv_inner.dilation,
                                                            group=conv_inner.group,
                                                            eps=bn_inner.eps,
                                                            momentum=1 - bn_inner.momentum,
                                                            has_bias=conv_inner.has_bias,
                                                            bias_init=conv_inner.bias_init,
                                                            quant_config=self.quant_config,
                                                            quant_dtype=self.weight_dtype)
                # change original network Batch Normalization OP parameters to quant network
                conv_inner.batchnorm.gamma = subcell.batchnorm.gamma
                conv_inner.batchnorm.beta = subcell.batchnorm.beta
                conv_inner.batchnorm.moving_mean = subcell.batchnorm.moving_mean
                conv_inner.batchnorm.moving_variance = subcell.batchnorm.moving_variance
            del subcell.batchnorm
            subcell.batchnorm = None
            subcell.has_bn = False
        else:
            conv_inner = quant.Conv2dQuant(conv_inner.in_channels, conv_inner.out_channels,
                                           kernel_size=conv_inner.kernel_size, stride=conv_inner.stride,
                                           pad_mode=conv_inner.pad_mode, padding=conv_inner.padding,
                                           dilation=conv_inner.dilation, group=conv_inner.group,
                                           has_bias=conv_inner.has_bias, quant_config=self.quant_config,
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
            subcell.activation = _AddFakeQuantAfterSubCell(F.identity, quant_dtype=self.act_dtype,
                                                           quant_delay=self.act_qdelay, per_channel=self.act_channel,
                                                           symmetric=self.act_symmetric, narrow_range=self.act_range,
                                                           optimize_option=self.optimize_option)
        return subcell

    def _convert_dense(self, subcell):
        """
        convert dense cell to quant cell
        """
        min_init = -6
        max_init = 6
        if OptimizeOption.LEARNED_SCALE in self.optimize_option:
            subcell_weight_para = subcell.dense.weight.data.asnumpy()
            if subcell.has_bn:
                scale_factor = (subcell.batchnorm.gamma.data.asnumpy() /
                                np.sqrt(subcell.batchnorm.moving_variance.data.asnumpy() + self.eps))
                subcell_weight_para = subcell_weight_para * scale_factor.reshape(-1, 1, 1, 1)
            min_init, max_init = self._kl_init(subcell_weight_para, self.weight_dtype)
        self.quant_config = self.quant_config._replace(
            weight=self.quant_config.weight.partial_init(min_init=min_init, max_init=max_init))

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
                                                           narrow_range=self.act_range,
                                                           optimize_option=self.optimize_option)
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

    def _kl_init(self, subcell_weight_para, weight_dtype):
        """
        Calculate the value of max_init and min_init with compute_kl_threshold.
        """
        if self.weight_channel:
            max_init = [compute_kl_threshold(weight_para_each, weight_dtype)
                        for weight_para_each in subcell_weight_para]
            min_init = [-x for x in max_init]
        else:
            max_init = [compute_kl_threshold(subcell_weight_para, weight_dtype)]
            min_init = [-x for x in max_init]
        return min_init, max_init

    def _set_mixed_bits(self, network, strategy):
        r"""
        Set network's quantization strategy, this function is currently only valid for `LEARNED_SCALE`
        optimize_option.

        Args:
            network (Cell): Input network.
            strategy (list): The quantization strategy for layers that need to be quantified (eg. [[8], [8],
                ..., [6], [4], [8]]), currently only the quant_dtype for weights of the dense layer and the
                convolution layer is supported.

        Returns:
            Cell, a network with mixed bit strategy configured.

        Raises:
            ValueError: If `OptimizeOption.LEARNED_SCALE` is not in `self.optimize_option`.
        """
        if OptimizeOption.LEARNED_SCALE not in self.optimize_option:
            raise ValueError("The `_set_mixed_bits` function is currently only valid for `LEARNED_SCALE` "
                             "optimize_option.")

        quantizable_idx = []
        pass_cell = None
        for i, cell_and_name in enumerate(network.cells_and_names()):
            cell = cell_and_name[1]
            if isinstance(cell, (nn.Conv2dBnAct, nn.DenseBnAct)) and cell is not pass_cell:
                quantizable_idx.append(i)

        if len(quantizable_idx) != len(strategy):
            raise ValueError("The dimension of quantifiable layers is not consistent with that of strategy.")

        quantizable_layer_bit_dict = {idx: bit for idx, bit in zip(quantizable_idx, strategy)}
        type_map = {
            QuantDtype.INT2.num_bits: QuantDtype.INT2,
            QuantDtype.INT3.num_bits: QuantDtype.INT3,
            QuantDtype.INT4.num_bits: QuantDtype.INT4,
            QuantDtype.INT5.num_bits: QuantDtype.INT5,
            QuantDtype.INT6.num_bits: QuantDtype.INT6,
            QuantDtype.INT7.num_bits: QuantDtype.INT7,
            QuantDtype.INT8.num_bits: QuantDtype.INT8
        }
        for i, cell_and_name in enumerate(network.cells_and_names()):
            cell = cell_and_name[1]
            if i not in quantizable_idx:
                continue
            else:
                if isinstance(cell, (nn.Conv2dBnAct, nn.DenseBnAct)):
                    cell.weight_dtype = type_map[quantizable_layer_bit_dict[i][0]]
                    if isinstance(cell, nn.Conv2dBnAct):
                        subcell_weight_para = cell.conv.weight.data.asnumpy()
                        if hasattr(cell.conv, 'gamma'):
                            scale_factor = (cell.conv.gamma.data.asnumpy() /
                                            np.sqrt(cell.conv.moving_variance.data.asnumpy() + self.eps))
                            subcell_weight_para = subcell_weight_para * scale_factor.reshape(-1, 1, 1, 1)
                        min_init, max_init = self._kl_init(subcell_weight_para, cell.weight_dtype)
                        cell.conv.fake_quant_weight.reset(quant_dtype=cell.weight_dtype,
                                                          min_init=min_init,
                                                          max_init=max_init)
                    elif isinstance(cell, nn.DenseBnAct):
                        subcell_weight_para = cell.dense.weight.data.asnumpy()
                        if hasattr(cell.dense, 'gamma'):
                            scale_factor = (cell.dense.gamma.data.asnumpy() /
                                            np.sqrt(cell.dense.moving_variance.data.asnumpy() + self.eps))
                            subcell_weight_para = subcell_weight_para * scale_factor.reshape(-1, 1, 1, 1)
                        min_init, max_init = self._kl_init(subcell_weight_para, cell.weight_dtype)
                        cell.dense.fake_quant_weight.reset(quant_dtype=cell.weight_dtype,
                                                           min_init=min_init,
                                                           max_init=max_init)
        return network
