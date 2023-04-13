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

"""Generate bprop for quantization aware ops"""

from mindspore.ops import operations as P
from mindspore.ops.operations import _quant_ops as Q
from mindspore.ops._grad_experimental.grad_base import bprop_getters
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore import context


@bprop_getters.register(Q.FakeQuantPerLayer)
def get_bprop_fakequant_with_minmax(self):
    """Generate bprop for FakeQuantPerLayer for GPU and Ascend"""
    op = Q.FakeQuantPerLayerGrad(
        num_bits=self.num_bits, quant_delay=self.quant_delay)

    def bprop(x, x_min, x_max, out, dout):
        dx = op(dout, x, x_min, x_max)
        return dx, zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(Q.FakeQuantWithMinMaxVars)
def get_bprop_fakequant_with_minmax_vars(self):
    """Generate bprop for FakeQuantWithMinMaxVars for Ascend"""
    op = Q.FakeQuantWithMinMaxVarsGradient(
        num_bits=self.num_bits, narrow_range=self.narrow_range)

    def bprop(x, x_min, x_max, out, dout):
        dx = op(dout, x, x_min, x_max)
        return dx, zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(Q.FakeQuantWithMinMaxVarsPerChannel)
def get_bprop_fakequant_with_minmax_vars_perchannel(self):
    """Generate bprop for FakeQuantWithMinMaxVarsPerChannel for Ascend"""
    op = Q.FakeQuantWithMinMaxVarsPerChannelGradient(
        num_bits=self.num_bits, narrow_range=self.narrow_range)

    def bprop(x, x_min, x_max, out, dout):
        dx = op(dout, x, x_min, x_max)
        return dx, zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(Q.FakeQuantPerChannel)
def get_bprop_fakequant_with_minmax_perchannel(self):
    """Generate bprop for FakeQuantPerChannel"""
    op = Q.FakeQuantPerChannelGrad(num_bits=self.num_bits,
                                   quant_delay=self.quant_delay,
                                   symmetric=self.symmetric,
                                   narrow_range=self.symmetric,
                                   channel_axis=self.channel_axis)

    def bprop(x, x_min, x_max, out, dout):
        dx = op(dout, x, x_min, x_max)
        return dx, zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(Q.BatchNormFold)
def get_bprop_batchnorm_fold(self):
    """Generate bprop for BatchNormFold for GPU"""
    op = Q.BatchNormFoldGrad(self.epsilon, self.is_training, self.freeze_bn)

    def bprop(x, mean, variance, global_step, out, dout):
        dx = op(dout[0], dout[1], x, out[0], out[1], global_step)
        res = (dx, zeros_like(mean), zeros_like(variance), zeros_like(global_step))
        return res

    return bprop


@bprop_getters.register(Q.CorrectionMul)
def get_bprop_correction_mul(self):
    """Generate bprop for CorrectionMul for Ascend and GPU"""
    grad_dx = Q.CorrectionMulGrad(self.channel_axis)
    grad_d_batch_std = Q.CorrectionMulGradReduce(self.channel_axis)

    def bprop(x, batch_std, running_std, out, dout):
        dx, d_batch_std = grad_dx(dout, x, batch_std, running_std)
        return dx, d_batch_std, zeros_like(running_std)

    def bprop_npu(x, batch_std, running_std, out, dout):
        dx, mul_dx = grad_dx(dout, x, batch_std, running_std)
        d_batch_std = grad_d_batch_std(mul_dx)
        return dx, d_batch_std, zeros_like(running_std)

    if context.get_context('device_target') == "Ascend":
        return bprop_npu

    return bprop


@bprop_getters.register(Q.BatchNormFold2)
def get_bprop_batchnorm_fold2(self):
    """Generate bprop for BatchNormFold2 for GPU"""
    op_f = Q.BatchNormFold2Grad(freeze_bn=self.freeze_bn)

    def bprop(x, beta, gamma, batch_std, batch_mean, running_std, running_mean, global_step, out, dout):
        d_batch_std, d_batch_mean, d_beta, d_gamma, d_x = op_f(dout, x, gamma, batch_std, batch_mean, running_std,
                                                               running_mean, global_step)
        res = (d_x, d_beta, d_gamma, d_batch_std, d_batch_mean, zeros_like(running_std), zeros_like(running_mean), \
            zeros_like(global_step))
        return res

    return bprop


@bprop_getters.register(Q.BatchNormFoldD)
def get_bprop_batchnormfold(self):
    """Generate bprop for BatchNormFold for Ascend"""
    op = Q.BatchNormFoldGradD(self.epsilon, self.is_training, self.freeze_bn)

    def bprop(x, x_sum, x_square_sum, mean, variance, out, dout):
        dx = op(dout[1], dout[2], x, out[1], out[2])
        res = (dx, zeros_like(x_sum), zeros_like(x_square_sum), zeros_like(mean), zeros_like(variance))
        return res

    return bprop


@bprop_getters.register(P.BNTrainingReduce)
def get_bprop_bn_training_reduce(self):
    """Generate bprop for BNTrainingReduce for Ascend"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(Q.BatchNormFold2D)
def get_bprop_batchnorm_fold2_(self):
    """Generate bprop for BatchNormFold2 for Ascend"""
    op_reduce = Q.BatchNormFold2GradReduce(freeze_bn=self.freeze_bn)
    op_f = Q.BatchNormFold2GradD(freeze_bn=self.freeze_bn)

    def bprop(x, beta, gamma, batch_std, batch_mean, running_std, out, dout):
        dout_reduce, dout_x_reduce = op_reduce(dout, x)
        d_batch_std, d_batch_mean, d_gamma, d_x = op_f(dout, dout_reduce, dout_x_reduce, gamma, batch_std,
                                                       batch_mean, running_std)
        res = (d_x, dout_reduce, d_gamma, d_batch_std, d_batch_mean, zeros_like(running_std))
        return res

    return bprop


@bprop_getters.register(Q.MinMaxUpdatePerLayer)
def get_bprop_fakequant_with_minmax_per_layer_update(self):
    """Generate bprop for MinMaxUpdatePerLayer for Ascend"""

    def bprop(x, x_min, x_max, out, dout):
        return zeros_like(x), zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(Q.MinMaxUpdatePerChannel)
def get_bprop_fakequant_with_minmax_per_channel_update(self):
    """Generate bprop for MinMaxUpdatePerChannel for Ascend"""

    def bprop(x, x_min, x_max, out, dout):
        return zeros_like(x), zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(Q.ActsULQ)
def get_bprop_acts_ulq(self):
    """Grad definition for 'ActsULQ' operation"""
    op = Q.ActsULQInputGrad()
    op1 = Q.ActULQClampMinGrad()
    op2 = Q.ActULQClampMaxGrad()

    def bprop(x, clamp_min, clamp_max, out, dout):
        dx = op(dout[0], out[1], out[2])
        dx1 = op1(dout[0], out[1], out[3])
        dx2 = op2(dout[0], out[2], out[3])
        return (dx, dx1, dx2)

    return bprop


@bprop_getters.register(Q.WtsARQ)
def get_bprop_wts_arq(self):
    """Grad definition for 'WtsArq' operation"""

    def bprop(w, w_min, w_max, out, dout):
        return (dout, zeros_like(w_min), zeros_like(w_max))

    return bprop


@bprop_getters.register(Q.FakeLearnedScaleQuantPerLayer)
def get_bprop_fakequant_with_learned_scale_perlayer(self):
    """Generate bprop for FakeLearnedScaleQuantPerLayer for GPU"""
    op = Q.FakeLearnedScaleQuantPerLayerGrad(quant_delay=self.quant_delay,
                                             neg_trunc=self.neg_trunc)

    def bprop(x, x_alpha, x_quant_max, out, dout):
        dx, dalpha = op(dout, x, x_alpha, x_quant_max)
        return dx, dalpha, zeros_like(x_quant_max)

    return bprop


@bprop_getters.register(Q.FakeLearnedScaleQuantPerChannel)
def get_bprop_fakequant_with_learned_scale_perchannel(self):
    """Generate bprop for FakeLearnedScaleQuantPerChannel for GPU"""
    op = Q.FakeLearnedScaleQuantPerChannelGrad(quant_delay=self.quant_delay,
                                               neg_trunc=self.neg_trunc,
                                               channel_axis=self.channel_axis)

    def bprop(x, x_alpha, x_quant_max, out, dout):
        dx, dalpha = op(dout, x, x_alpha, x_quant_max)
        return dx, dalpha, zeros_like(x_quant_max)

    return bprop
