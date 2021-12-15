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
from ...common.api import _cell_graph_executor as _executor
from ...common.parameter import Parameter
from ...nn import Cell
from ...nn.layer import quant
from ...ops import operations as P
from ...ops import functional as F
from ...ops.operations import _inner_ops as inner
from ..quant import quant_utils
from ..quant.qat import _AddFakeQuantInput, _AddFakeQuantAfterSubCell


__all__ = ["ExportToQuantInferNetwork"]


class QuantBlock(Cell):
    r"""
    A quant block of Conv/Dense, activation layer for Ascend deploy.

    Calculate Conv or Dense in Int8, with Quant and DeQuant.

    Notes:
        This block is only for deploy, and not trainable.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): The regularization function applied to the output of the layer, eg. 'relu'. Default: None.
        batchnorm (bool): Specifies to used batchnorm or not. Default: None.
        activation (string): Specifies activation type. The optional values are as following:
            'softmax', 'logsoftmax', 'relu', 'relu6', 'tanh', 'gelu', 'sigmoid',
            'prelu', 'leakyrelu', 'hswish', 'hsigmoid'. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`.
    """

    def __init__(self,
                 core_op,
                 weight,
                 quant_op,
                 dequant_op,
                 dequant_scale,
                 bias=None,
                 activation=None):
        super(QuantBlock, self).__init__()
        self.core_op = core_op
        self.weight = weight
        self.quant = quant_op
        self.dequant = dequant_op
        self.dequant_scale = dequant_scale
        self.bias = bias
        self.has_bias = bias is not None
        self.activation = activation
        self.has_act = activation is not None
        self.bias_add = P.BiasAdd()
        self.sub = P.Sub()
        self.weight_offset = Parameter(np.zeros(1, dtype=np.int8), name='weight_offset')

    def construct(self, x):
        x = self.quant(x)
        if self.has_bias:
            weight = self.sub(self.weight, self.weight_offset)
            x = self.core_op(x, weight)
            x = self.bias_add(x, self.bias)
        else:
            x = self.core_op(x, self.weight)
        x = self.dequant(x, self.dequant_scale)
        x = F.cast(x, mstype.float32)
        if self.has_act:
            x = self.activation(x)
        return x

    def extend_repr(self):
        s = f'quant={self.quant}, core_op={type(self.core_op)}, weight=shape[{self.weight.shape}]'
        if self.has_bias:
            s += f', bias=shape[{self.bias.shape}]'
        if self.has_act:
            s += f', activation={self.activation}'
        s += f', dequant={self.dequant}'
        return s


class QuantMindirBlock(Cell):
    """A quant binary block of Conv/Dense, activation layer for export MINDIR model.

       Args:
        core_op (Cell): The operation cell.
        weight (Tensor): The weight of the cell.
        bias (Tensor): The bias of the cell. Default: None.
        activation (str): The regularization function applied to the output of the layer, eg. 'relu'. Default: None.
        param_dict (dict): The information of the cell.
    """

    def __init__(self,
                 core_op,
                 weight,
                 bias=None,
                 activation=None,
                 param_dict=None):

        super(QuantMindirBlock, self).__init__()
        self.core_op = core_op
        if activation is not None:
            self.core_op.add_prim_attr("activation_name", activation.__class__.__name__)
        self.core_op.add_prim_attr("filter_maxq", Tensor(param_dict["filter_maxq"]))
        self.core_op.add_prim_attr("filter_minq", Tensor(param_dict["filter_minq"]))
        if param_dict["output_maxq"] is not None:
            self.core_op.add_prim_attr("output_maxq", Tensor(param_dict["output_maxq"]))
            self.core_op.add_prim_attr("output_minq", Tensor(param_dict["output_minq"]))
        self.core_op.add_prim_attr("symmetric", Tensor(param_dict["symmetric"]))
        if hasattr(core_op, 'pad_mode'):
            self.core_op.add_prim_attr("pad_mode", core_op.pad_mode)
        self.core_op.add_prim_attr("act_num_bits", Tensor(8))
        self.core_op.add_prim_attr("weight_num_bits", Tensor(param_dict["weight_num_bits"]))
        self.core_op.add_prim_attr("weight_narrow_range", Tensor(param_dict["weight_narrow_range"]))
        if param_dict["input_narrow_range"] is not None:
            self.core_op.add_prim_attr("input_narrow_range", Tensor(param_dict["input_narrow_range"]))
        if param_dict["output_narrow_range"] is not None:
            self.core_op.add_prim_attr("output_narrow_range", Tensor(param_dict["output_narrow_range"]))
        if param_dict["input_maxq"] == 'None':
            self.core_op.add_prim_attr("mean", Tensor(param_dict["mean"]))
            self.core_op.add_prim_attr("std_dev", Tensor(param_dict["std_dev"]))
        elif param_dict["input_maxq"] is not None:
            self.core_op.add_prim_attr("input_maxq", Tensor(param_dict["input_maxq"]))
            self.core_op.add_prim_attr("input_minq", Tensor(param_dict["input_minq"]))

        self.weight = weight
        self.bias = bias
        self.has_bias = bias is not None
        self.activation = activation
        self.has_act = activation is not None
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        if self.has_bias:
            x = self.core_op(x, self.weight)
            x = self.bias_add(x, self.bias)
        else:
            x = self.core_op(x, self.weight)
        if self.has_act:
            x = self.activation(x)
        return x

    def extend_repr(self):
        s = f'core_op={type(self.core_op)}, weight=shape[{self.weight.shape}]'
        if self.has_bias:
            s += f', bias=shape[{self.bias.shape}]'
        if self.has_act:
            s += f', activation={self.activation}'
        return s


class ExportToQuantInferNetwork:
    """
    Convert quantization aware network to infer network.

    Args:
        network (Cell): MindSpore quantization aware training network.
        inputs (Tensor): Input tensors of the `quantization aware training network`.
        mean (int, float): The mean of input data after preprocessing, used for quantizing the first layer of network.
          Default: 127.5.
        std_dev (int, float): The variance of input data after preprocessing, used for quantizing the first layer
          of network. Default: 127.5.
        is_mindir (bool): Whether export MINDIR format. Default: False.

    Returns:
        Cell, Infer network.
    """

    def __init__(self, network, mean, std_dev, *inputs, is_mindir=False):
        network = Validator.check_isinstance('network', network, (nn.Cell,))
        self.data_type = mstype.int8
        self.network = copy.deepcopy(network)
        self.network_bk = copy.deepcopy(network)
        self.get_inputs_table(inputs)
        self.mean = mean
        self.std_dev = std_dev
        self.is_mindir = is_mindir
        self.upcell = None

    def get_inputs_table(self, inputs):
        """Get the input quantization parameters of quantization cell for quant export."""
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
        scale_a_in, zp_a_in, scale_w, zp_w, param_dict = self.__get_quant_param(cell_core, fake_quant_a_out)

        # Build the `Quant` `Dequant` op.
        # Quant only support perlayer version. Need check here.
        quant_op = inner.Quant(1 / float(scale_a_in), float(zp_a_in))
        scale_deq = self.__get_dequant_scale(scale_a_in, scale_w)
        dequant_op = inner.Dequant()

        if isinstance(activation, _AddFakeQuantAfterSubCell):
            activation = activation.subcell
        elif hasattr(activation, "get_origin"):
            activation = activation.get_origin()

        # get op
        if isinstance(cell_core, quant.DenseQuant):
            op_core = P.MatMul()
        else:
            op_core = cell_core.conv

        # get the `weight` and `bias`
        weight, bias, weight_b, bias_b = self.__get_weight_bias(cell_core, scale_a_in, scale_w, zp_w)

        if self.is_mindir:
            block = QuantMindirBlock(op_core, weight_b, bias_b, activation, param_dict)
        else:
            block = QuantBlock(op_core, weight, quant_op, dequant_op, scale_deq, bias, activation)
        return block

    def _get_input_quant_param(self, minq_name, np_type, param_dict):
        """get input quant parameter for quant block"""
        fake_quant_a_in_prefix = minq_name[:-5]
        cells = self.network_bk.cells_and_names()
        for cell in cells:
            if cell[0].endswith(fake_quant_a_in_prefix):
                fake_quant_a_in = cell[1]
                break
        scale_a_in, zp_a_in, param_dict["input_maxq"], param_dict["input_minq"] = \
            quant_utils.scale_zp_max_min_from_fake_quant_cell(fake_quant_a_in, np_type)
        param_dict["input_narrow_range"] = fake_quant_a_in.narrow_range
        return scale_a_in, zp_a_in

    def __get_quant_param(self, cell_core, fake_quant_a_out):
        """get parameter for quant block"""
        w_minq_name = cell_core.fake_quant_weight.minq.name
        w_maxq_name = cell_core.fake_quant_weight.maxq.name
        np_type = mstype.dtype_to_nptype(self.data_type)
        param_dict = dict()
        param_dict["filter_maxq"] = None
        param_dict["filter_minq"] = None
        param_dict["output_maxq"] = None
        param_dict["output_minq"] = None
        param_dict["input_maxq"] = None
        param_dict["input_minq"] = None
        param_dict["input_narrow_range"] = None
        param_dict["output_narrow_range"] = None
        param_dict["weight_narrow_range"] = cell_core.fake_quant_weight.narrow_range
        param_dict["mean"] = self.mean
        param_dict["std_dev"] = self.std_dev
        param_dict["symmetric"] = cell_core.fake_quant_weight.symmetric
        param_dict["weight_num_bits"] = cell_core.fake_quant_weight.num_bits

        scale_w, zp_w, param_dict["filter_maxq"], param_dict["filter_minq"] = \
            quant_utils.scale_zp_max_min_from_fake_quant_cell(cell_core.fake_quant_weight, np_type)
        if fake_quant_a_out is not None:
            _, _, param_dict["output_maxq"], param_dict["output_minq"] = \
                quant_utils.scale_zp_max_min_from_fake_quant_cell(fake_quant_a_out, np_type)
            param_dict["output_narrow_range"] = fake_quant_a_out.narrow_range

        info = self.quant_info_table.get(w_minq_name, None)
        if not info:
            info = self.quant_info_table.get(w_maxq_name, None)
        if info:
            _, minq_name = info
            if minq_name == 'input':
                scale_a_in, zp_a_in, param_dict["input_maxq"], param_dict["input_minq"] = \
                    (1 / self.std_dev), round(self.mean), 'None', 'None'
            else:
                scale_a_in, zp_a_in = self._get_input_quant_param(minq_name, np_type, param_dict)
        else:
            # skip quant layer
            scale_a_in, zp_a_in = 1.0, 0.0
        return scale_a_in, zp_a_in, scale_w, zp_w, param_dict

    @staticmethod
    def __get_dequant_scale(scale_a_in, scale_w):
        """Get dequant scale"""
        scale_deq = scale_a_in * scale_w

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
        return scale_deq

    def __get_weight_bias(self, cell_core, scale_a_in, scale_w, zp_w):
        """Get weight and bias for quantizaiton"""
        np_type = mstype.dtype_to_nptype(self.data_type)
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
        quant_min, quant_max = quant_utils.get_quant_min_max(np_type,
                                                             cell_core.fake_quant_weight.num_bits,
                                                             cell_core.fake_quant_weight.narrow_range)
        weight = quant_utils.weight2int(weight, scale_w, zp_w, quant_min, quant_max)
        if bias is not None:
            bias = Tensor(bias / scale_a_in / scale_w, mstype.int32)

        if isinstance(cell_core, quant.DenseQuant):
            weight = np.transpose(weight)
            weight_b = np.transpose(weight_b)

        weight = Tensor(weight, self.data_type)
        weight_b = Tensor(weight_b)
        if bias_b is not None:
            bias_b = Tensor(bias_b, mstype.float32)
        return weight, bias, weight_b, bias_b

    def _add_output_min_max_for_op(self, origin_op, fake_quant_cell):
        """add output quant info for quant op for export mindir."""
        if self.is_mindir:
            if isinstance(origin_op, ops.Primitive) and not hasattr(origin_op, 'output_minq'):
                np_type = mstype.dtype_to_nptype(self.data_type)
                _, _, maxq, minq = quant_utils.scale_zp_max_min_from_fake_quant_cell(fake_quant_cell, np_type)
                origin_op.add_prim_attr('output_maxq', Tensor(maxq))
                origin_op.add_prim_attr('output_minq', Tensor(minq))

    def _convert_subcell(self, network, change, name, subcell):
        """Convert subcell to ant subcell."""
        if subcell is not None and hasattr(subcell, "fake_quant_weight"):
            new_subcell = self._get_quant_block(subcell, None, None)
            prefix = subcell.param_prefix
            new_subcell.update_parameters_name(prefix + '.')
            self.upcell = new_subcell
            network.insert_child_to_cell(name, new_subcell)
            change = True
        return network, change

    def _convert_conv(self, network, change, name, subcell):
        """Convert subcell to ant subcell for conv."""
        cell_core = subcell.conv
        activation = subcell.activation
        fake_quant_act = None
        if hasattr(activation, 'fake_quant_act_before'):
            fake_quant_act = activation.fake_quant_act_before
        elif hasattr(activation, 'fake_quant_act'):
            fake_quant_act = activation.fake_quant_act
        if cell_core is not None and hasattr(cell_core, "fake_quant_weight"):
            new_subcell = self._get_quant_block(cell_core, activation, fake_quant_act)
            self.upcell = None
            prefix = subcell.param_prefix
            new_subcell.update_parameters_name(prefix + '.')
            network.insert_child_to_cell(name, new_subcell)
            change = True
        return network, change

    def _convert_dense(self, network, change, name, subcell):
        """Convert subcell to ant subcell for dense."""
        cell_core = subcell.dense
        activation = subcell.activation
        fake_quant_act = None
        if hasattr(activation, 'fake_quant_act_before'):
            fake_quant_act = activation.fake_quant_act_before
        elif hasattr(activation, 'fake_quant_act'):
            fake_quant_act = activation.fake_quant_act
        if cell_core is not None and hasattr(cell_core, "fake_quant_weight"):
            new_subcell = self._get_quant_block(cell_core, activation, fake_quant_act)
            prefix = subcell.param_prefix
            new_subcell.update_parameters_name(prefix + '.')
            network.insert_child_to_cell(name, new_subcell)
            self.upcell = None
            change = True
        return network, change

    def _convert_act(self, subcell):
        """Convert subcell to ant subcell for activation."""
        activation = subcell.get_origin()
        if isinstance(activation, nn.ReLU):
            self._add_output_min_max_for_op(activation.relu, subcell.fake_quant_act)
        elif isinstance(activation, nn.ReLU6):
            self._add_output_min_max_for_op(activation.relu6, subcell.fake_quant_act)
        if self.upcell:
            self._add_output_min_max_for_op(self.upcell.core_op, subcell.fake_quant_act)
        return activation

    def _convert_add(self, subcell):
        """Convert subcell to ant subcell for add."""
        if isinstance(subcell.add, _AddFakeQuantAfterSubCell):
            add_op = subcell.add.subcell
            subcell.__delattr__("add")
            subcell.__setattr__("add", add_op)
        add_op = subcell.add
        self._add_output_min_max_for_op(add_op, subcell.fake_quant_act)
        subcell.__delattr__("fake_quant_act")
        subcell.__setattr__("fake_quant_act", P.identity())

    def _convert_observer(self, network, name, subcell):
        """Convert subcell to ant subcell for FakeQuantWithMinMaxObserver."""
        if self.upcell:
            self._add_output_min_max_for_op(self.upcell.core_op, subcell)
        network.__delattr__(name)
        network.__setattr__(name, P.identity())

    def _convert_fake_quant_after_cell(self, network, name, subcell):
        """Convert subcell to ant subcell for _AddFakeQuantAfterSubCell."""
        op = subcell.subcell
        self._add_output_min_max_for_op(op, subcell.fake_quant_act)
        network.__delattr__(name)
        network.__setattr__(name, op)

    def _convert_core_quant_subcell(self, network, change, name, subcell):
        """Convert subcell to ant subcell for conv and dense."""
        is_core_subcell = True
        if isinstance(subcell, nn.Conv2dBnAct):
            network, change = self._convert_conv(network, change, name, subcell)
        elif isinstance(subcell, nn.DenseBnAct):
            network, change = self._convert_dense(network, change, name, subcell)
        elif isinstance(subcell, (quant.Conv2dBnFoldQuant, quant.Conv2dBnFoldQuantOneConv,
                                  quant.Conv2dBnWithoutFoldQuant, quant.Conv2dQuant, quant.DenseQuant)):
            network, change = self._convert_subcell(network, change, name, subcell)
        else:
            is_core_subcell = False
        return is_core_subcell, network, change

    def _convert_other_quant_subcell(self, network, change, name, subcell):
        """Convert subcell to ant subcell for cell except conv and dense."""
        is_other_subcell = True
        if isinstance(subcell, nn.ActQuant) and hasattr(subcell, "get_origin"):
            activation = self._convert_act(subcell)
            network.insert_child_to_cell(name, activation)
            change = True
        elif isinstance(subcell, nn.TensorAddQuant):
            self._convert_add(subcell)
        elif isinstance(subcell, quant.FakeQuantWithMinMaxObserver):
            self._convert_observer(network, name, subcell)
        elif isinstance(subcell, _AddFakeQuantAfterSubCell):
            self._convert_fake_quant_after_cell(network, name, subcell)
            change = True
        else:
            is_other_subcell = False
        return is_other_subcell, network, change

    def _convert_quant2deploy(self, network):
        """Convert network's all quant subcell to deploy subcell."""
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            is_core_quant_subcell, network, change = self._convert_core_quant_subcell(network, change, name, subcell)
            is_other_quant_subcell, network, change = self._convert_other_quant_subcell(network, change, name, subcell)
            if not is_core_quant_subcell and not is_other_quant_subcell:
                self.upcell = None
                self._convert_quant2deploy(subcell)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())
        return network
