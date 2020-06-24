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
"""quantization aware."""

import copy
import re

import numpy as np
import mindspore.context as context

from ... import log as logger
from ... import nn, ops
from ..._checkparam import ParamValidator as validator
from ..._checkparam import Rel
from ...common import Tensor
from ...common import dtype as mstype
from ...common.api import _executor
from ...nn.layer import quant
from ...ops import functional as F
from ...ops.operations import _inner_ops as inner
from ...train import serialization
from . import quant_utils


_ACTIVATION_MAP = {nn.ReLU: quant.ReLUQuant,
                   nn.ReLU6: quant.ReLU6Quant,
                   nn.HSigmoid: quant.HSigmoidQuant,
                   nn.HSwish: quant.HSwishQuant}


class _AddFakeQuantInput(nn.Cell):
    """
    Add FakeQuant at input and output of the Network. Only support one input and one output case.
    """

    def __init__(self, network, quant_delay=0):
        super(_AddFakeQuantInput, self).__init__(auto_prefix=False)
        self.network = network
        self.fake_quant_input = quant.FakeQuantWithMinMax(
            min_init=-6, max_init=6, quant_delay=quant_delay, ema=True)
        self.fake_quant_input.update_parameters_name('fake_quant_input')

    def construct(self, data):
        data = self.fake_quant_input(data)
        output = self.network(data)
        return output


class _AddFakeQuantAfterSubCell(nn.Cell):
    """
    Add FakeQuant after of the sub Cell.
    """

    def __init__(self, subcell, **kwargs):
        super(_AddFakeQuantAfterSubCell, self).__init__(auto_prefix=False)
        self.subcell = subcell
        self.fake_quant_act = quant.FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        ema=True,
                                                        num_bits=kwargs["num_bits"],
                                                        quant_delay=kwargs["quant_delay"],
                                                        per_channel=kwargs["per_channel"],
                                                        symmetric=kwargs["symmetric"],
                                                        narrow_range=kwargs["narrow_range"])

    def construct(self, *data):
        output = self.subcell(*data)
        output = self.fake_quant_act(output)
        return output


class ConvertToQuantNetwork:
    """
    Convert network to quantization aware network
    """
    __quant_op_name__ = ["TensorAdd", "Sub", "Mul", "RealDiv"]

    def __init__(self, **kwargs):
        self.network = validator.check_isinstance('network', kwargs["network"], (nn.Cell,))
        self.weight_qdelay = validator.check_integer("quant delay", kwargs["quant_delay"][0], 0, Rel.GE)
        self.act_qdelay = validator.check_integer("quant delay", kwargs["quant_delay"][-1], 0, Rel.GE)
        self.bn_fold = validator.check_bool("bn fold", kwargs["bn_fold"])
        self.freeze_bn = validator.check_integer("freeze bn", kwargs["freeze_bn"], 0, Rel.GE)
        self.weight_bits = validator.check_integer("weights bit", kwargs["num_bits"][0], 0, Rel.GE)
        self.act_bits = validator.check_integer("activations bit", kwargs["num_bits"][-1], 0, Rel.GE)
        self.weight_channel = validator.check_bool("per channel", kwargs["per_channel"][0])
        self.act_channel = validator.check_bool("per channel", kwargs["per_channel"][-1])
        self.weight_symmetric = validator.check_bool("symmetric", kwargs["symmetric"][0])
        self.act_symmetric = validator.check_bool("symmetric", kwargs["symmetric"][-1])
        self.weight_range = validator.check_bool("narrow range", kwargs["narrow_range"][0])
        self.act_range = validator.check_bool("narrow range", kwargs["narrow_range"][-1])
        self._convert_method_map = {quant.Conv2dBnAct: self._convert_conv,
                                    quant.DenseBnAct: self._convert_dense}

    def _convert_op_name(self, name):
        pattern = re.compile(r'([A-Z]{1})')
        name_new = re.sub(pattern, r'_\1', name).lower()
        if name_new[0] == '_':
            name_new = name_new[1:]
        return name_new

    def run(self):
        self.network.update_cell_prefix()
        network = self._convert_subcells2quant(self.network)
        network = _AddFakeQuantInput(network)
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
            elif isinstance(subcell, (quant.Conv2dBnAct, quant.DenseBnAct)):
                prefix = subcell.param_prefix
                new_subcell = self._convert_method_map[type(subcell)](subcell)
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
            add_quant = _AddFakeQuantAfterSubCell(prim_op,
                                                  num_bits=self.act_bits,
                                                  quant_delay=self.act_delay,
                                                  per_channel=self.act_channel,
                                                  symmetric=self.act_symmetric,
                                                  narrow_range=self.act_range)
            prefix = '.'.join([network.param_prefix, self._convert_op_name(prim_op.name)])
            add_quant.update_parameters_name(prefix + '.')
            del network.__dict__[name]
            network.insert_child_to_cell(name, add_quant)
        return network

    def _convert_conv(self, subcell):
        """
        convet conv cell to quant cell
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
                                                    quant_delay=self.weight_qdelay,
                                                    freeze_bn=self.freeze_bn,
                                                    per_channel=self.weight_channel,
                                                    num_bits=self.weight_bits,
                                                    fake=True,
                                                    symmetric=self.weight_symmetric,
                                                    narrow_range=self.weight_range)
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
                                           quant_delay=self.weight_qdelay,
                                           per_channel=self.weight_channel,
                                           num_bits=self.weight_bits,
                                           symmetric=self.weight_symmetric,
                                           narrow_range=self.weight_range)
        subcell.conv = conv_inner
        if subcell.has_act and subcell.activation is not None:
            subcell.activation = self._convert_activation(subcell.activation)
        else:
            subcell.has_act = True
            subcell.activation = _AddFakeQuantAfterSubCell(F.identity,
                                                           num_bits=self.act_bits,
                                                           quant_delay=self.act_qdelay,
                                                           per_channel=self.act_channel,
                                                           symmetric=self.act_symmetric,
                                                           narrow_range=self.act_range)
        return subcell

    def _convert_dense(self, subcell):
        """
        convert dense cell to combine dense cell
        """
        dense_inner = subcell.dense
        dense_inner = quant.DenseQuant(dense_inner.in_channels,
                                       dense_inner.out_channels,
                                       has_bias=dense_inner.has_bias,
                                       num_bits=self.weight_bits,
                                       quant_delay=self.weight_qdelay,
                                       per_channel=self.weight_channel,
                                       symmetric=self.weight_symmetric,
                                       narrow_range=self.weight_range)
        subcell.dense = dense_inner
        if subcell.has_act and subcell.activation is not None:
            subcell.activation = self._convert_activation(subcell.activation)
        else:
            subcell.has_act = True
            subcell.activation = _AddFakeQuantAfterSubCell(F.identity,
                                                           num_bits=self.act_bits,
                                                           quant_delay=self.act_qdelay,
                                                           per_channel=self.act_channel,
                                                           symmetric=self.act_symmetric,
                                                           narrow_range=self.act_range)
        return subcell

    def _convert_activation(self, activation):
        act_class = activation.__class__
        if act_class not in _ACTIVATION_MAP:
            raise ValueError(
                "Unsupported activation in auto Quant: ", act_class)
        return _ACTIVATION_MAP[act_class](num_bits=self.act_bits,
                                          quant_delay=self.act_qdelay,
                                          per_channel=self.act_channel,
                                          symmetric=self.weight_symmetric,
                                          narrow_range=self.weight_range)


class ExportQuantNetworkDeploy:
    """
    Convert quantization aware network to deploy network.

    Args:
        network (Cell): MindSpore network produced by `convert_quant_network`.
        inputs (Tensor): Inputs of the `network`.

    Returns:
        Cell, converted network.
    """
    __quant_op_name__ = ["TensorAdd", "Sub", "Mul", "RealDiv"]

    def __init__(self,
                 network,
                 *inputs):
        network = validator.check_isinstance('network', network, (nn.Cell,))
        self.data_type = mstype.int8
        self.network = copy.deepcopy(network)
        self.all_paramters = {p.name: p for p in self.network.get_parameters()}
        self.get_inputs_table(inputs)

    def get_inputs_table(self, inputs):
        """Get the support info for quant export."""
        phase_name = 'export_quant'
        graph_id, _ = _executor.compile(self.network, *inputs, phase=phase_name, do_convert=False)
        self.quant_info_table = _executor.fetch_info_for_quant_export(graph_id)

    def run(self):
        """Start to convert."""
        self.network.update_cell_prefix()
        network = self.network
        if isinstance(network, _AddFakeQuantInput):
            network = network.network
        network = self._convert_quant2deploy(network)
        return network

    def _get_quant_block(self, cell_core, activation, fake_quant_a_out):
        """convet network's quant subcell to deploy subcell"""
        # Calculate the scale and zero point
        w_minq_name = cell_core.fake_quant_weight.minq.name
        np_type = mstype.dtype_to_nptype(self.data_type)
        scale_w, zp_w = quant_utils.scale_zp_from_fack_quant_cell(cell_core.fake_quant_weight, np_type)
        scale_a_out, _ = quant_utils.scale_zp_from_fack_quant_cell(fake_quant_a_out, np_type)
        info = self.quant_info_table.get(w_minq_name, None)
        if info:
            fack_quant_a_in_op, minq_name = info
            maxq = self.all_paramters[minq_name[:-4] + "maxq"]
            minq = self.all_paramters[minq_name]
            scale_a_in, zp_a_in = quant_utils.scale_zp_from_data(fack_quant_a_in_op, maxq, minq, np_type)
        else:
            logger.warning(f"Do not find `fake_quant` from input with `fack_quant.minq` {w_minq_name}")
            return None

        # Build the `Quant` `Dequant` op.
        # AscendQuant only support perlayer version. Need check here.
        quant_op = inner.AscendQuant(float(scale_a_in), float(zp_a_in))
        sqrt_mode = False
        scale_deq = scale_a_out * scale_w
        if scale_deq < 2 ** -14:
            scale_deq = np.sqrt(scale_deq)
            sqrt_mode = True
        dequant_op = inner.AscendDequant(sqrt_mode)

        # get op
        op_core = cell_core.matmul if isinstance(cell_core, quant.DenseQuant) else cell_core.conv
        if isinstance(activation, _AddFakeQuantAfterSubCell):
            activation = activation.subcell
        elif hasattr(activation, "get_origin"):
            activation = activation.get_origin()

        # get the `weight` and `bias`
        weight = cell_core.weight.data.asnumpy()
        bias = None
        if isinstance(cell_core, (quant.DenseQuant, quant.Conv2dQuant)):
            if cell_core.has_bias:
                bias = cell_core.bias.data.asnumpy()
        elif isinstance(cell_core, quant.Conv2dBatchNormQuant):
            weight, bias = quant_utils.fold_batchnorm(weight, cell_core)

        # apply the quant
        weight = Tensor(quant_utils.weight2int(weight, scale_w, zp_w), self.data_type)
        if bias is not None:
            bias = Tensor(scale_a_in * scale_w * bias, mstype.int32)
        scale_deq = Tensor(scale_deq, mstype.float16)
        block = quant.QuantBlock(op_core, weight, quant_op, dequant_op, scale_deq, bias, activation)
        return block

    def _convert_quant2deploy(self, network):
        """Convet network's all quant subcell to deploy subcell."""
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            cell_core = None
            fake_quant_act = None
            activation = None
            if isinstance(subcell, quant.Conv2dBnAct):
                cell_core = subcell.conv
                activation = subcell.activation
                fake_quant_act = activation.fake_quant_act
            elif isinstance(subcell, quant.DenseBnAct):
                cell_core = subcell.dense
                activation = subcell.activation
                fake_quant_act = activation.fake_quant_act
            if cell_core is not None:
                new_subcell = self._get_quant_block(cell_core, activation, fake_quant_act)
                if new_subcell:
                    prefix = subcell.param_prefix
                    new_subcell.update_parameters_name(prefix + '.')
                    network.insert_child_to_cell(name, new_subcell)
                    change = True
            elif isinstance(subcell, _AddFakeQuantAfterSubCell):
                op = subcell.subcell
                if op.name in ConvertToQuantNetwork.__quant_op_name__ and isinstance(op, ops.Primitive):
                    network.__delattr__(name)
                    network.__setattr__(name, op)
                    change = True
            else:
                self._convert_quant2deploy(subcell)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())
        return network


def export_geir(network, *inputs, file_name):
    """
    Exports MindSpore quant predict model to deploy with GEIR.

    Args:
        network (Cell): MindSpore network produced by `convert_quant_network`.
        inputs (Tensor): Inputs of the `network`.
        file_name (str): File name of model to export.
    """
    exporter = ExportQuantNetworkDeploy(network, *inputs)
    deploy_net = exporter.run()
    serialization.export(deploy_net, *inputs, file_name=file_name, file_format="GEIR")


def convert_quant_network(network,
                          bn_fold=False,
                          freeze_bn=0,
                          quant_delay=(0, 0),
                          num_bits=(8, 8),
                          per_channel=(False, False),
                          symmetric=(False, False),
                          narrow_range=(False, False)
                          ):
    r"""
    Create quantization aware training network.

    Args:
        network (Cell): Obtain a pipeline through network for saving graph summary.
        quant_delay (int or tuple): Number of steps after which weights and activations are quantized during
            eval. The first element represent weights and second element represent data flow. Default: (0, 0)
        bn_fold (bool): Flag to used bn fold ops for simulation inference operation. Default: False.
        freeze_bn (int): Number of steps after which BatchNorm OP parameters used total mean and variance. Default: 0.
        num_bits (int or tuple): Number of bits to use for quantizing weights and activations. The first
            element represent weights and second element represent data flow. Default: (8, 8)
        per_channel (int or tuple):  Quantization granularity based on layer or on channel. If `True`
            then base on per channel otherwise base on per layer. The first element represent weights
            and second element represent data flow. Default: (False, False)
        symmetric (int or tuple): Quantization algorithm use symmetric or not. If `True` then base on
            symmetric otherwise base on assymmetric. The first element represent weights and second
            element represent data flow. Default: (False, False)
        narrow_range (int or tuple): Quantization algorithm use narrow range or not. If `True` then base
            on narrow range otherwise base on off narrow range. The first element represent weights and
            second element represent data flow. Default: (False, False)

    Returns:
        Cell, Network which has change to quantization aware training network cell.
    """
    support_device = ["Ascend", "GPU"]
    def convert2list(name, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            value = [value]
        elif len(value) > 2:
            raise ValueError("input `{}` len should less then 2".format(name))
        return value

    quant_delay = convert2list("quant delay", quant_delay)
    num_bits = convert2list("num bits", num_bits)
    per_channel = convert2list("per channel", per_channel)
    symmetric = convert2list("symmetric", symmetric)
    narrow_range = convert2list("narrow range", narrow_range)

    if context.get_context('device_target') not in support_device:
        raise KeyError("Not support {} backend.".format(context.get_context('device_target')))

    net = ConvertToQuantNetwork(network=network,
                                quant_delay=quant_delay,
                                bn_fold=bn_fold,
                                freeze_bn=freeze_bn,
                                num_bits=num_bits,
                                per_channel=per_channel,
                                symmetric=symmetric,
                                narrow_range=narrow_range)
    return net.run()
