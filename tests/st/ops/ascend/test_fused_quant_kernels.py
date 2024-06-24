# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
from mindspore import Parameter, Tensor, context, dtype, GRAPH_MODE, JitConfig, PYNATIVE_MODE
from mindspore.nn import Cell
from mindspore.ops import operations as msops
from mindspore.ops.operations._inner_ops import Quant
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul


class NumpyQuantOps:
    """
    numpy quant ops for test.
    """
    @staticmethod
    def anti_quant(data, scale, offset, sqrt_mode=False, dst_type=np.float16):
        """
        convert compressed dtype to orin dtype
        anti_quant_data = (data - offset) * scale (* scale if sqrt_mode is True)
        """
        anti_quant_data = data.astype(np.float32)
        if sqrt_mode:
            return ((anti_quant_data - offset) * scale * scale).astype(dst_type)
        return ((anti_quant_data - offset) * scale).astype(dst_type)

    @staticmethod
    def quant(data, scale, offset, sqrt_mode=False, dst_type=np.int8):
        """
        compress data to lower bit dtype
        quant_data = data / scale + offset
        """
        if sqrt_mode:
            quant_data = np.round(data / (scale * scale) + offset)
        else:
            quant_data = np.round(data / scale + offset)
        return quant_data.astype(dst_type)

    @staticmethod
    def trans_fp32_to_u64(scale_fp32):
        """transport fp32 data to uint64"""
        fp32_scale_deq = np.array(scale_fp32, dtype=np.float32)
        ui32_scale_deq = np.frombuffer(fp32_scale_deq, np.uint32)
        ui64_scale_deq = np.zeros(fp32_scale_deq.shape, np.uint64)
        ui64_scale_deq |= np.uint64(ui32_scale_deq)
        return ui64_scale_deq.tolist()


class NumpyFullQuant:
    """full quant process using numpy"""
    def __init__(self,
                 weight_scale,
                 act_scale,
                 act_offset):
        self.weight_scale = weight_scale
        self.act_scale = act_scale
        self.act_offset = act_offset

    def process(self, activation, weight, bias):
        """ numpy implementation of A8W8 matmul with bias correction"""
        quant_act = NumpyQuantOps.quant(activation, self.act_scale, self.act_offset)
        quant_weight = NumpyQuantOps.quant(weight, self.weight_scale, 0)
        quant_bias = (bias / (self.act_scale * self.weight_scale)).astype(np.int32)
        fused_bias = -np.sum(self.act_offset.astype(np.int32) * quant_weight.astype(np.int32), axis=0) + quant_bias
        quant_result = np.matmul(quant_act.astype(np.int32), quant_weight.astype(np.int32)) + fused_bias
        dequant_result = quant_result * self.act_scale * self.weight_scale
        return dequant_result


class QuantDequantCell(Cell):
    """matmul and dequant fused cell"""
    def __init__(self,
                 weight,
                 weight_scale,
                 act_scale,
                 act_offset,
                 bias):
        super().__init__()
        dst_dtype = dtype.float16
        self.dbmm = QuantBatchMatmul(transpose_x1=False,
                                     transpose_x2=False,
                                     dtype=dst_dtype)
        self.dequant_scale = self._dequant_scale(act_scale, weight_scale)
        self.act_scale = act_scale
        self.act_offset = act_offset
        self.weight_scale = weight_scale

        t_scale = 1.0 / act_scale
        self.quant = Quant(t_scale.tolist()[0], act_offset.tolist()[0])

        self.quant_weight = Tensor(NumpyQuantOps.quant(weight, weight_scale, 0), dtype=dtype.int8)
        self.bias = Tensor(self._fused_bias(bias), dtype=dtype.int32)

    def _dequant_scale(self, act_scale, weight_scale):
        """calculate dequant scale"""
        dequant_scale = weight_scale * act_scale
        scale_ui64 = NumpyQuantOps.trans_fp32_to_u64(dequant_scale)
        return Parameter(Tensor(np.squeeze(scale_ui64), dtype=dtype.uint64))

    def _fused_bias(self, bias):
        """fused bias correction into bias"""
        bias_int32 = (bias / (self.act_scale * self.weight_scale)).astype(np.int32)
        add_item = - np.sum(self.act_offset.astype(np.int32) * self.quant_weight.asnumpy().astype(np.int32),
                            axis=0).astype(np.int32)
        return bias_int32 + add_item

    def construct(self, x):
        """construct quant and dequant forward process"""
        quant_act = self.quant(x)
        # (matmul(quant_act, x2) + bias) * scale + offset
        return self.dbmm(quant_act, self.quant_weight, self.dequant_scale, None, self.bias)


class AntiquantBMMCell(Cell):
    """fused anti quant cell."""
    def __init__(self,
                 scale,
                 offset,
                 out_dtype=dtype.float16,
                 transpose_x: bool = False,
                 transpose_weight: bool = False):
        super().__init__()
        self.out_dtype = out_dtype
        self.scale = Parameter(Tensor(np.squeeze(scale), dtype=self.out_dtype))
        self.zp_neg = Parameter(Tensor(np.squeeze(np.array(offset)) * -1, dtype=self.out_dtype))
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight)
        self.cast = msops.Cast()

    def construct(self,
                  x,
                  weight,
                  bias=None):
        """forward for antiquant bmm cell"""
        out = self.weight_qbmm(x, weight, self.scale, self.zp_neg, None, None, bias)
        return self.cast(out, self.out_dtype)


class DequantBMMCell(Cell):
    """matmul and dequant fused cell"""

    def __init__(self,
                 scale,
                 offset=None,
                 transpose_a=False,
                 transpose_b=False,
                 dst_dtype=dtype.float16):
        super().__init__()
        self.dbmm = QuantBatchMatmul(transpose_x1=transpose_a,
                                     transpose_x2=transpose_b,
                                     dtype=dst_dtype)
        scale_ui64 = NumpyQuantOps.trans_fp32_to_u64(scale)
        self.scale = Parameter(Tensor(np.squeeze(scale_ui64), dtype=dtype.uint64))
        if offset is None:
            self.offset = Parameter(Tensor(np.zeros(self.scale.shape), dtype=dtype.float32))
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    def construct(self, x1, x2, bias):
        """dequant bmm forward procedure"""
        # (matmul(x1, x2) + bias) * scale + offset
        return self.dbmm(x1, x2, self.scale, self.offset, bias)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_weight_quant_bmm_cell_as_antiquant_1p(mode):
    """
    Feature: weight quant bmm cell for antiquant
    Description: test antiquant using weight quant bmm cell
    Expectation: accuracy in tolerance
    """

    weight = np.array([[100, 200, 100], [10, 25, 10]]).astype(np.int8)
    activation = np.array([[0.1, 1., 0.1], [0.5, 2.4, 0.5]]).astype(np.float16)
    scale = np.array([0.5, 0.27]).astype(np.float16)
    offset = np.array([-127, -10]).astype(np.float16)
    expect = np.matmul(activation, NumpyQuantOps.anti_quant(np.transpose(weight), scale, offset))
    wqmm_cell = AntiquantBMMCell(scale, offset, dtype.float16, False, True)
    if mode == 'KBK':
        context.set_context(device_target="Ascend", mode=GRAPH_MODE)
        wqmm_cell.set_jit_config(JitConfig(jit_level='O0'))
    elif mode == 'GE':
        context.set_context(device_target="Ascend", mode=GRAPH_MODE)
    else:
        context.set_context(device_target="Ascend", mode=PYNATIVE_MODE)
    t_activation = Tensor(activation, dtype=dtype.float16)
    p_weight = Parameter(Tensor(weight, dtype=dtype.int8), 'weight')
    fact = wqmm_cell(t_activation, p_weight).asnumpy()
    np.testing.assert_allclose(fact, expect, rtol=3e-2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_quant_batch_matmul_with_bias_correction(mode):
    """
    Feature: test quant and dequant cell with bias correction
    Description: test quant and dequant procedure correction
    Expectation: accuracy in tolerance
    """
    weight = np.array([[2., 4.], [1., 3.]]).astype(np.float16)
    activation = np.array([[1, 10.], [12, 14]]).astype(np.float16)
    weight_scale = np.array([0.5, 0.7]).astype(np.float16)
    act_offset = np.array([10]).astype(np.float16)
    activation_scale = np.array([0.5], dtype=np.float16)
    bias = np.ones([2]).astype(np.float16)

    net = NumpyFullQuant(weight_scale, activation_scale, act_offset)
    quant_out = net.process(activation, weight, bias)
    cell = QuantDequantCell(weight,
                            weight_scale,
                            activation_scale,
                            act_offset,
                            bias)
    if mode == 'KBK':
        context.set_context(device_target="Ascend", mode=GRAPH_MODE)
        cell.set_jit_config(JitConfig(jit_level='O0'))
    elif mode == 'GE':
        context.set_context(device_target="Ascend", mode=GRAPH_MODE)
    else:
        context.set_context(device_target="Ascend", mode=PYNATIVE_MODE)

    t_activation = Tensor(activation, dtype=dtype.float16)
    ms_quant_out = cell(t_activation).asnumpy()
    np.testing.assert_allclose(quant_out, ms_quant_out, rtol=7e-2)
