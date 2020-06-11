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
"""aware quantization."""

import re
from ... import nn
from ... import ops
from ..._checkparam import ParamValidator as validator
from ..._checkparam import Rel
from ...nn.layer import combined
from ...nn.layer import quant

_ACTIVATION_MAP = {nn.ReLU: quant.ReLUQuant,
                   nn.ReLU6: quant.ReLU6Quant,
                   nn.HSigmoid: quant.HSigmoidQuant,
                   nn.HSwish: quant.HSwishQuant}


class _AddFakeQuantInputOutput(nn.Cell):
    """
    Add FakeQuant at input and output of the Network. Only support one input and one output case.
    """

    def __init__(self, network, quant_delay=0):
        super(_AddFakeQuantInputOutput, self).__init__(auto_prefix=False)
        self.network = network
        self.fake_quant_input = quant.FakeQuantWithMinMax(
            min_init=-6, max_init=6, quant_delay=quant_delay, ema=True)
        self.fake_quant_input.update_parameters_name('fake_quant_input')
        self.fake_quant_output = quant.FakeQuantWithMinMax(
            min_init=-6, max_init=6, quant_delay=quant_delay, ema=True)
        self.fake_quant_output.update_parameters_name('fake_quant_output')

    def construct(self, data):
        data = self.fake_quant_input(data)
        output = self.network(data)
        output = self.fake_quant_output(output)
        return output


class _AddFakeQuantAfterSubCell(nn.Cell):
    """
    Add FakeQuant after of the sub Cell.
    """

    def __init__(self, subcell, quant_delay=0, num_bits=8):
        super(_AddFakeQuantAfterSubCell, self).__init__(auto_prefix=False)
        self.subcell = subcell
        self.fake_quant_act = quant.FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        num_bits=num_bits,
                                                        quant_delay=quant_delay,
                                                        ema=True)

    def construct(self, *data):
        output = self.subcell(*data)
        output = self.fake_quant_act(output)
        return output


class ConvertToQuantNetwork:
    """
    Convert network to quantization aware network
    """
    __quant_op_name__ = ["TensorAdd", "Sub", "Mul", "RealDiv"]

    def __init__(self,
                 network,
                 quant_delay=0,
                 bn_fold=False,
                 freeze_bn=0,
                 weight_bits=8,
                 act_bits=8,
                 per_channel=False,
                 symmetric=False,
                 narrow_range=False):
        self.network = validator.check_isinstance(
            'network', network, (nn.Cell,))
        self.quant_delay = validator.check_integer(
            "quant delay", quant_delay, 0, Rel.GE)
        self.freeze_bn = validator.check_integer(
            "freeze bn", freeze_bn, 0, Rel.GE)
        self.weight_bits = validator.check_integer(
            "weights bit", weight_bits, 0, Rel.GE)
        self.act_bits = validator.check_integer(
            "activations bit", act_bits, 0, Rel.GE)
        self.bn_fold = validator.check_bool("bn fold", bn_fold)
        self.per_channel = validator.check_bool("per channel", per_channel)
        self.symmetric = validator.check_bool("symmetric", symmetric)
        self.narrow_range = validator.check_bool("narrow range", narrow_range)

    def _convert_op_name(self, name):
        pattern = re.compile(r'([A-Z]{1})')
        name_new = re.sub(pattern, r'_\1', name).lower()
        if name_new[0] == '_':
            name_new = name_new[1:]
        return name_new

    def run(self):
        self.network.update_cell_prefix()
        network = self._convert_subcells2quant(self.network)
        return network

    def _convert_subcells2quant(self, network):
        """
        convet sub cell to quant cell
        """
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            elif isinstance(subcell, combined.Conv2d):
                prefix = subcell.param_prefix
                new_subcell = self._convert_conv(subcell)
                new_subcell.update_parameters_name(prefix + '.')
                network.insert_child_to_cell(name, new_subcell)
                change = True
            elif isinstance(subcell, combined.Dense):
                prefix = subcell.param_prefix
                new_subcell = self._convert_dense(subcell)
                new_subcell.update_parameters_name(prefix + '.')
                network.insert_child_to_cell(name, new_subcell)
                change = True
            else:
                self._convert_subcells2quant(subcell)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())

        # tensoradd to tensoradd quant
        add_list = []
        for name in network.__dict__:
            if name[0] == '_':
                continue
            attr = network.__dict__[name]
            if isinstance(attr, ops.Primitive) and attr.name in ConvertToQuantNetwork.__quant_op_name__:
                add_list.append((name, attr))
        for name, prim_op in add_list:
            prefix = name
            add_quant = _AddFakeQuantAfterSubCell(prim_op)  # quant.TensorAddQuant()
            prefix = '.'.join([network.param_prefix, self._convert_op_name(prim_op.name)])
            add_quant.update_parameters_name(prefix + '.')
            del network.__dict__[name]
            network.insert_child_to_cell(name, add_quant)
        return network

    def _convert_conv(self, subcell):
        """
        convet conv cell to combine cell
        """
        conv_inner = subcell.conv
        bn_inner = subcell.batchnorm
        if subcell.batchnorm is not None and self.bn_fold:
            conv_inner = quant.Conv2dBatchNormQuant(conv_inner.in_channels,
                                                    conv_inner.out_channels,
                                                    kernel_size=conv_inner.kernel_size,
                                                    stride=conv_inner.stride,
                                                    pad_mode=conv_inner.pad_mode,
                                                    padding=conv_inner.padding,
                                                    dilation=conv_inner.dilation,
                                                    group=conv_inner.group,
                                                    eps=bn_inner.eps,
                                                    momentum=bn_inner.momentum,
                                                    quant_delay=self.quant_delay,
                                                    freeze_bn=self.freeze_bn,
                                                    per_channel=self.per_channel,
                                                    num_bits=self.weight_bits,
                                                    fake=True,
                                                    symmetric=self.symmetric,
                                                    narrow_range=self.narrow_range)
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
                                           quant_delay=self.quant_delay,
                                           per_channel=self.per_channel,
                                           num_bits=self.weight_bits,
                                           symmetric=self.symmetric,
                                           narrow_range=self.narrow_range)
        subcell.conv = conv_inner
        if subcell.activation is not None:
            subcell.activation = self._convert_activation(subcell.activation)
        else:
            subcell = _AddFakeQuantAfterSubCell(subcell)
        return subcell

    def _convert_dense(self, subcell):
        """
        convert dense cell to combine dense cell
        """
        dense_inner = subcell.dense
        dense_inner = quant.DenseQuant(dense_inner.in_channels,
                                       dense_inner.out_channels,
                                       has_bias=dense_inner.has_bias,
                                       quant_delay=self.quant_delay,
                                       per_channel=self.per_channel,
                                       num_bits=self.weight_bits)
        subcell.dense = dense_inner
        if subcell.activation is not None:
            subcell.activation = self._convert_activation(subcell.activation)
        return subcell

    def _convert_activation(self, activation):
        act_class = activation.__class__
        if act_class not in _ACTIVATION_MAP:
            raise ValueError(
                "Unsupported activation in auto Quant: ", act_class)
        return _ACTIVATION_MAP[act_class](num_bits=self.act_bits, quant_delay=self.quant_delay)


def convert_quant_network(network,
                          quant_delay=0,
                          bn_fold=False,
                          freeze_bn=0,
                          weight_bits=8,
                          act_bits=8,
                          per_channel=False,
                          symmetric=False,
                          narrow_range=False
                          ):
    r"""
    Create aware quantizaiton training network.

    Args:
        network (Cell): Obtain a pipeline through network for saving graph summary.
        quant_delay (int): Number of steps after which weights and activations are quantized during eval. Default: 0.
        bn_fold (bool): Flag to used bn fold ops for simulation inference operation. Default: False.
        freeze_bn (int): Number of steps after which BN parameters used total mean and variance. Default: 0.
        weight_bits (int): Number of bits to use for quantizing weights. Default: 8.
        act_bits (int): Number of bits to use for quantizing activations. Default: 8.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    returns:
        Cell, Network which has change to aware quantization training network.
    """
    net = ConvertToQuantNetwork(
        network, quant_delay, bn_fold, freeze_bn, weight_bits, act_bits, per_channel, symmetric, narrow_range)
    return net.run()
