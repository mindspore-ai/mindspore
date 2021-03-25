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
"""Export for quantization."""

import copy

import numpy as np

from ... import nn, ops
from ..._checkparam import Validator
from ...common import Tensor
from ...common import dtype as mstype
from ...common.api import _executor
from ...nn.layer import quant
from ...ops import operations as P
from ...ops.operations import _inner_ops as inner
from ..quant import quant_utils
from ..quant.qat import QuantizationAwareTraining, _AddFakeQuantInput, _AddFakeQuantAfterSubCell


__all__ = ["ExportToQuantInferNetwork", "ExportManualQuantNetwork"]

class ExportToQuantInferNetwork:
    """
    Convert quantization aware network to infer network.

    Args:
        network (Cell): MindSpore network API `convert_quant_network`.
        inputs (Tensor): Input tensors of the `quantization aware training network`.
        mean (int): Input data mean. Default: 127.5.
        std_dev (int, float): Input data variance. Default: 127.5.
        is_mindir (bool): Whether is MINDIR format. Default: False.

    Returns:
        Cell, Infer network.
    """
    __quant_op_name__ = ["Add", "Sub", "Mul", "RealDiv"]

    def __init__(self, network, mean, std_dev, *inputs, is_mindir=False):
        network = Validator.check_isinstance('network', network, (nn.Cell,))
        self.input_scale = 1 / std_dev
        self.input_zero_point = round(mean)
        self.data_type = mstype.int8
        self.network = copy.deepcopy(network)
        self.all_parameters = {p.name: p for p in self.network.get_parameters()}
        self.get_inputs_table(inputs)
        self.mean = mean
        self.std_dev = std_dev
        self.is_mindir = is_mindir

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
        """convert network's quant subcell to deploy subcell"""
        # Calculate the scale and zero point
        w_minq_name = cell_core.fake_quant_weight.minq.name
        np_type = mstype.dtype_to_nptype(self.data_type)
        param_dict = dict()
        param_dict["filter_maxq"] = None
        param_dict["filter_minq"] = None
        param_dict["output_maxq"] = None
        param_dict["output_minq"] = None
        param_dict["input_maxq"] = None
        param_dict["input_minq"] = None
        param_dict["mean"] = self.mean
        param_dict["std_dev"] = self.std_dev
        param_dict["symmetric"] = cell_core.fake_quant_weight.symmetric

        scale_w, zp_w, param_dict["filter_maxq"], param_dict["filter_minq"] = \
            quant_utils.scale_zp_max_min_from_fake_quant_cell(cell_core.fake_quant_weight, np_type)
        if fake_quant_a_out is not None:
            _, _, param_dict["output_maxq"], param_dict["output_minq"] = \
                quant_utils.scale_zp_max_min_from_fake_quant_cell(fake_quant_a_out, np_type)

        info = self.quant_info_table.get(w_minq_name, None)
        if info:
            fake_quant_a_in_op, minq_name = info
            if minq_name == 'input':
                scale_a_in, zp_a_in, param_dict["input_maxq"], param_dict["input_minq"] = \
                    self.input_scale, self.input_zero_point, 'None', 'None'
            else:
                maxq = self.all_parameters[minq_name[:-4] + "maxq"]
                minq = self.all_parameters[minq_name]
                scale_a_in, zp_a_in, param_dict["input_maxq"], param_dict["input_minq"] = \
                    quant_utils.scale_zp_max_min_from_data(fake_quant_a_in_op, minq, maxq, np_type)
        else:
            # skip quant layer
            scale_a_in, zp_a_in = 1.0, 0.0

        # Build the `Quant` `Dequant` op.
        # Quant only support perlayer version. Need check here.
        quant_op = inner.Quant(1 / float(scale_a_in), float(zp_a_in))
        scale_deq = scale_a_in * scale_w
        dequant_op = inner.Dequant()

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
        elif isinstance(cell_core, (quant.Conv2dBnFoldQuant, quant.Conv2dBnFoldQuantOneConv)):
            weight, bias = quant_utils.fold_batchnorm(weight, cell_core)
        elif isinstance(cell_core, quant.Conv2dBnWithoutFoldQuant):
            weight, bias = quant_utils.without_fold_batchnorm(weight, cell_core)
        weight_b = weight
        bias_b = bias
        # apply the quant
        fake_quant_weight_op = cell_core.fake_quant_weight.fake_quant_infer
        weight = quant_utils.weight2int(weight, scale_w, zp_w, np_type, fake_quant_weight_op.num_bits,
                                        fake_quant_weight_op.narrow_range)
        if bias is not None:
            bias = Tensor(bias / scale_a_in / scale_w, mstype.int32)

        # fuse parameter
        # |--------|47:40|--------|39:32|--------|31:0|
        #         offset_w [8]    shift_N [8]    deq_scale [32]
        float32_deq_scale = scale_deq.astype(np.float32)
        uint32_deq_scale = np.frombuffer(float32_deq_scale, np.uint32)
        scale_length = scale_deq.size  # channel
        dequant_param = np.zeros(scale_length, dtype=np.uint64)
        for index in range(scale_length):
            dequant_param[index] += uint32_deq_scale[index]

        scale_deq = Tensor(dequant_param, mstype.uint64)
        # get op
        if isinstance(cell_core, quant.DenseQuant):
            op_core = P.MatMul()
            weight = np.transpose(weight)
            weight_b = np.transpose(weight_b)
        else:
            op_core = cell_core.conv
        weight = Tensor(weight, self.data_type)
        weight_b = Tensor(weight_b)
        if bias_b is not None:
            bias_b = Tensor(bias_b, mstype.float32)
        if self.is_mindir:
            block = quant.QuantMindirBlock(op_core, weight_b, bias_b, activation, param_dict)
        else:
            block = quant.QuantBlock(op_core, weight, quant_op, dequant_op, scale_deq, bias, activation)
        return block

    def _convert_quant2deploy(self, network):
        """Convert network's all quant subcell to deploy subcell."""
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            cell_core = None
            fake_quant_act = None
            activation = None
            if isinstance(subcell, nn.Conv2dBnAct):
                cell_core = subcell.conv
                activation = subcell.activation
                fake_quant_act = activation.fake_quant_act if hasattr(activation, "fake_quant_act") else None
            elif isinstance(subcell, nn.DenseBnAct):
                cell_core = subcell.dense
                activation = subcell.activation
                fake_quant_act = activation.fake_quant_act if hasattr(activation, "fake_quant_act") else None
            if cell_core is not None:
                new_subcell = self._get_quant_block(cell_core, activation, fake_quant_act)
                if new_subcell:
                    prefix = subcell.param_prefix
                    new_subcell.update_parameters_name(prefix + '.')
                    network.insert_child_to_cell(name, new_subcell)
                    change = True
            elif isinstance(subcell, _AddFakeQuantAfterSubCell):
                op = subcell.subcell
                if op.name in QuantizationAwareTraining.__quant_op_name__ and isinstance(op, ops.Primitive):
                    if self.is_mindir:
                        op.add_prim_attr('output_maxq', Tensor(subcell.fake_quant_act.maxq.data.asnumpy()))
                        op.add_prim_attr('output_minq', Tensor(subcell.fake_quant_act.minq.data.asnumpy()))
                    network.__delattr__(name)
                    network.__setattr__(name, op)
                    change = True
            else:
                self._convert_quant2deploy(subcell)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())
        return network

class ExportManualQuantNetwork(ExportToQuantInferNetwork):
    """
        Convert manual quantization aware network to infer network.

        Args:
            network (Cell): MindSpore network API `convert_quant_network`.
            inputs (Tensor): Input tensors of the `quantization aware training network`.
            mean (int): Input data mean. Default: 127.5.
            std_dev (int, float): Input data variance. Default: 127.5.
            is_mindir (bool): Whether is MINDIR format. Default: False.

        Returns:
            Cell, Infer network.
        """
    __quant_op_name__ = ["Add", "Sub", "Mul", "RealDiv"]

    def __init__(self, network, mean, std_dev, *inputs, is_mindir=False):
        super(ExportManualQuantNetwork, self).__init__(network, mean, std_dev, *inputs, is_mindir)
        self.upcell = None
        self.upname = None

    def _convert_quant2deploy(self, network):
        """Convert network's all quant subcell to deploy subcell."""
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            if isinstance(subcell, nn.Conv2dBnAct):
                network, change = self._convert_subcell(network, change, name, subcell)
            elif isinstance(subcell, nn.DenseBnAct):
                network, change = self._convert_subcell(network, change, name, subcell, conv=False)
            elif isinstance(subcell, (quant.Conv2dBnFoldQuant, quant.Conv2dBnWithoutFoldQuant,
                                      quant.Conv2dQuant, quant.DenseQuant)):
                network, change = self._convert_subcell(network, change, name, subcell, core=False)
            elif isinstance(subcell, quant.FakeQuantWithMinMaxObserver) and self.upcell:
                np_type = mstype.dtype_to_nptype(self.data_type)
                _, _, maxq, minq = quant_utils.scale_zp_max_min_from_fake_quant_cell(subcell, np_type)
                self.upcell.core_op.add_prim_attr('output_maxq', Tensor(maxq))
                self.upcell.core_op.add_prim_attr('output_minq', Tensor(minq))
                network.insert_child_to_cell(self.upname, self.upcell)
            elif isinstance(subcell, _AddFakeQuantAfterSubCell):
                op = subcell.subcell
                if op.name in QuantizationAwareTraining.__quant_op_name__ and isinstance(op, ops.Primitive):
                    if self.is_mindir:
                        op.add_prim_attr('output_maxq', Tensor(subcell.fake_quant_act.maxq.data.asnumpy()))
                        op.add_prim_attr('output_minq', Tensor(subcell.fake_quant_act.minq.data.asnumpy()))
                    network.__delattr__(name)
                    network.__setattr__(name, op)
                    change = True
            else:
                self.upcell, self.upname = None, None
                self._convert_quant2deploy(subcell)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())
        return network

    def _convert_subcell(self, network, change, name, subcell, core=True, conv=True):
        """Convert subcell to ant subcell."""
        if core:
            cell_core = subcell.conv if conv else subcell.dense
            activation = subcell.activation
            fake_quant_act = activation.fake_quant_act
        else:
            cell_core = subcell
            activation = None
            fake_quant_act = None
        new_subcell = self._get_quant_block(cell_core, activation, fake_quant_act)
        if new_subcell:
            prefix = subcell.param_prefix
            new_subcell.update_parameters_name(prefix + '.')
            self.upcell = None if core else new_subcell
            self.upname = None if core else name
            network.insert_child_to_cell(name, new_subcell)
            change = True
        return network, change
