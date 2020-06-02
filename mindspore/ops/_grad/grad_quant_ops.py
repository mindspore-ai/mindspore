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

"""Generate bprop for aware quantization ops"""

from .. import operations as P
from .grad_base import bprop_getters
from ..composite.multitype_ops.zeros_like_impl import zeros_like


@bprop_getters.register(P.FakeQuantWithMinMax)
def get_bprop_fakequant_with_minmax(self):
    """Generate bprop for FakeQuantWithMinMax for GPU and Ascend"""
    op = P.FakeQuantWithMinMaxGrad(num_bits=self.num_bits, quant_delay=self.quant_delay)

    def bprop(x, x_min, x_max, out, dout):
        dx = op(dout, x, x_min, x_max)
        return dx, zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(P.FakeQuantWithMinMaxPerChannel)
def get_bprop_fakequant_with_minmax_perchannel(self):
    """Generate bprop for FakeQuantWithMinMaxPerChannel for GPU"""
    op = P.FakeQuantWithMinMaxPerChannelGrad(num_bits=self.num_bits, quant_delay=self.quant_delay)

    def bprop(x, x_min, x_max, out, dout):
        dx = op(dout, x, x_min, x_max)
        return dx, zeros_like(x_min), zeros_like(x_max)

    return bprop


@bprop_getters.register(P.BatchNormFold)
def get_bprop_batchnorm_fold(self):
    """Generate bprop for BatchNormFold for GPU"""
    op = P.BatchNormFoldGrad(self.epsilon, self.is_training, self.freeze_bn)

    def bprop(x, mean, variance, global_step, out, dout):
        dx = op(dout[0], dout[1], x, out[0], out[1], global_step)
        return dx, zeros_like(mean), zeros_like(variance), zeros_like(global_step)

    return bprop


@bprop_getters.register(P.CorrectionMul)
def get_bprop_correction_mul(self):
    """Generate bprop for CorrectionMul for Ascend and GPU"""
    grad = P.CorrectionMulGrad(self.channel_axis)

    def bprop(x, batch_std, running_std, out, dout):
        dx, d_batch_std = grad(dout, x, batch_std, running_std)
        return dx, d_batch_std, zeros_like(running_std)

    return bprop


@bprop_getters.register(P.BatchNormFold2)
def get_bprop_batchnorm_fold2(self):
    """Generate bprop for BatchNormFold2 for GPU"""
    op_f = P.BatchNormFold2Grad(freeze_bn=self.freeze_bn)

    def bprop(x, beta, gamma, batch_std, batch_mean, running_std, running_mean, global_step, out, dout):
        d_batch_std, d_batch_mean, d_beta, d_gamma, d_x = op_f(dout, x, gamma, batch_std, batch_mean, running_std,
                                                               running_mean, global_step)
        return d_x, d_beta, d_gamma, d_batch_std, d_batch_mean, zeros_like(running_std), zeros_like(running_mean), \
               zeros_like(global_step)

    return bprop


@bprop_getters.register(P.BatchNormFoldD)
def get_bprop_BatchNormFold(self):
    """Generate bprop for BatchNormFold for Ascend"""
    op = P.BatchNormFoldGradD(self.epsilon, self.is_training, self.freeze_bn)

    def bprop(x, x_sum, x_square_sum, mean, variance, out, dout):
        dx = op(dout[1], dout[2], x, out[1], out[2])
        return dx, zeros_like(x_sum), zeros_like(x_square_sum), zeros_like(mean), zeros_like(variance)

    return bprop


@bprop_getters.register(P.BNTrainingReduce)
def get_bprop_BNTrainingReduce(self):
    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.BatchNormFold2_D)
def get_bprop_batchnorm_fold2_(self):
    """Generate bprop for BatchNormFold2 for Ascend"""
    op_reduce = P.BatchNormFold2GradReduce(freeze_bn=self.freeze_bn)
    op_f = P.BatchNormFold2GradD(freeze_bn=self.freeze_bn)

    def bprop(x, beta, gamma, batch_std, batch_mean, running_std, out, dout):
        dout_reduce, dout_x_reduce = op_reduce(dout, x)
        d_batch_std, d_batch_mean, d_gamma, d_x = op_f(dout, dout_reduce, dout_x_reduce, gamma, batch_std,
                                                       batch_mean, running_std)
        return d_x, dout_reduce, d_gamma, d_batch_std, d_batch_mean, zeros_like(running_std)

    return bprop


@bprop_getters.register(P.FakeQuantWithMinMaxUpdate)
def get_bprop_fakequant_with_minmax_update(self):
    """Generate bprop for FakeQuantWithMinMaxUpdate for Ascend"""

    def bprop(x, x_min, x_max, out, dout):
        return zeros_like(x), zeros_like(x_min), zeros_like(x_max)

    return bprop
