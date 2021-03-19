# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Generate vm_impl function for nn ops"""
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.vm_impl_registry import vm_impl_registry as vm_impl_getters
from .vm_interface import vm


# pylint: disable=unused-argument


@vm_impl_getters.register(P.ScalarSummary)
def vm_impl_scalar_summary(self):
    """Generate vm_impl function for ScalarSummary"""

    def vm_impl(string_in, scalar):
        """Implement by vm mode."""
        return scalar

    return vm_impl


@vm_impl_getters.register(P.ReLU)
def vm_impl_relu(self):
    """Generate vm_impl function for ReLU"""

    def vm_impl(x):
        x = x.asnumpy()
        output = Tensor(vm.relu(x))
        return output

    return vm_impl


@vm_impl_getters.register(P.Flatten)
def vm_impl_flatten(self):
    """Generate vm_impl function for Flatten"""

    def vm_impl(x):
        x = x.asnumpy()
        return Tensor(vm.flatten_batch(x))

    return vm_impl


@vm_impl_getters.register(P.Softmax)
def vm_impl_softmax(self):
    """Generate vm_impl function for Softmax"""

    def vm_impl(x):
        x = x.asnumpy()
        return Tensor(vm.softmax(x))

    return vm_impl


@vm_impl_getters.register(P.LogSoftmax)
def vm_impl_log_softmax(self):
    """Generate vm_impl function for LogSoftmax"""

    def vm_impl(x):
        x = x.asnumpy()
        return Tensor(vm.logsoftmax(x))

    return vm_impl


@vm_impl_getters.register(P.Tanh)
def vm_impl_tanh(self):
    """Generate vm_impl function for Tanh"""

    def vm_impl(x):
        x = x.asnumpy()
        return Tensor(vm.tanh(x))

    return vm_impl


@vm_impl_getters.register(P.BatchNorm)
def vm_impl_batch_norm(self):
    """Generate vm_impl function for BatchNorm"""

    def vm_impl(x, scale, b, mean, variance):
        # pylint: disable=unused-argument
        x = x.asnumpy()
        scale = scale.asnumpy()
        b = b.asnumpy()
        mean = mean.asnumpy()
        variance = variance.asnumpy()
        out, x_mean, x_var, running_mean, running_var = vm.batch_norm(x, scale, b, mean, \
                                                                      variance, \
                                                                      eps=self.epsilon)
        return Tensor(out), Tensor(x_mean), Tensor(x_var), \
               Tensor(running_mean), Tensor(running_var)

    return vm_impl


@vm_impl_getters.register(P.Conv2D)
def vm_impl_conv2d(self):
    """Generate vm_impl function for Conv2D"""

    def vm_impl(x, w):
        x = x.asnumpy()
        weight = w.asnumpy()
        bias = None
        out = vm.conv2d(x, weight, bias, self.stride, self.pad, self.dilation)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(G.MaxPoolGradWithArgmax)
def vm_impl_max_pool_grad_with_argmax(self):
    """Generate vm_impl function for MaxPoolGradWithArgmax"""

    def vm_impl(x, dout, argmax):
        x = x.asnumpy()
        dout = dout.asnumpy()
        arg_max = argmax.asnumpy()
        dx = vm.max_pool_grad_with_argmax(x, dout, arg_max,
                                          self.kernel_size[1], self.kernel_size[2], self.strides[1])
        return Tensor(dx)

    return vm_impl


@vm_impl_getters.register(P.MaxPoolWithArgmax)
def vm_impl_max_pool_with_argmax(self):
    """Generate vm_impl function for MaxPoolWithArgmax"""

    def vm_impl(x):
        x = x.asnumpy()
        out, out_argmax = vm.max_pool_with_argmax(x, self.kernel_size[1], self.kernel_size[2], self.strides[1])
        return Tensor(out), Tensor(out_argmax)

    return vm_impl


@vm_impl_getters.register(P.MaxPool)
def vm_impl_max_pool(self):
    """Generate vm_impl function for MaxPool"""

    def vm_impl(x):
        x = x.asnumpy()
        out = vm.max_pooling(x, self.kernel_size[-2], self.kernel_size[-1], self.strides[-2])
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(G.MaxPoolGrad)
def vm_impl_max_pool_grad(self):
    """Generate vm_impl function for MaxPoolGrad"""

    def vm_impl(x, out, dout):
        x = x.asnumpy()
        dout = dout.asnumpy()
        out = vm.max_pool_grad(x, dout, self.kernel_size[-2], self.kernel_size[-1], self.strides[-2])
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.AvgPool)
def vm_impl_avg_pool(self):
    """Generate vm_impl function for AvgPool"""

    def vm_impl(x):
        x = x.asnumpy()
        out = vm.avg_pooling(x, self.kernel_size[-2], self.kernel_size[-1], self.strides[-2])
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(G.AvgPoolGrad)
def vm_impl_avg_pool_grad(self):
    """Generate vm_impl function for AvgPoolGrad"""

    def vm_impl(dout, origin_shape):
        dout = dout.asnumpy()
        out = vm.avg_pool_grad(dout, origin_shape, self.kernel_size[-2], self.kernel_size[-1], self.strides[-2])
        return Tensor(out)

    return vm_impl


# pylint: disable=function-redefined
@vm_impl_getters.register(G.BatchNormGrad)
def vm_impl_fused_batch_norm_grad(self):
    """Generate vm_impl function for BatchNormGrad"""

    def vm_impl(dy, x, scale, save_mean, save_inv_variance):
        dy = dy.asnumpy()
        x = x.asnumpy()
        scale = scale.asnumpy()
        save_mean = save_mean.asnumpy()
        save_inv_variance = save_inv_variance.asnumpy()
        dx, dscale, dshift = vm.batch_norm_grad(dy, x, scale, save_mean, save_inv_variance)
        return (Tensor(dx), Tensor(dscale), Tensor(dshift))

    return vm_impl


@vm_impl_getters.register(G.ReluGrad)
def vm_impl_relu_grad(self):
    """Generate vm_impl function for ReluGrad"""

    def vm_impl(y_backprop, x):
        x = x.asnumpy()
        y_backprop = y_backprop.asnumpy()
        y_backprop = vm.relu_grad(x.copy()) * y_backprop
        return Tensor(y_backprop)

    return vm_impl


@vm_impl_getters.register(P.Conv2DBackpropInput)
def vm_impl_conv2d_backprop_input(self):
    """Generate vm_impl function for Conv2DBackpropInput"""

    def vm_impl(dout, w, x_size):
        dout = dout.asnumpy()
        w = w.asnumpy()
        dx = vm.conv2d_backprop_input(dout, x_size, w, self.stride, self.pad)
        return Tensor(dx)

    return vm_impl


@vm_impl_getters.register(G.Conv2DBackpropFilter)
def vm_impl_conv2d_backprop_filter(self):
    """Generate vm_impl function for Conv2DBackpropFilter"""

    def vm_impl(dout, x, w_size):
        x = x.asnumpy()
        dout = dout.asnumpy()
        dw = vm.conv2d_backprop_filter(dout, x, w_size, self.stride, self.pad)
        return Tensor(dw)

    return vm_impl


@vm_impl_getters.register(G.FlattenGrad)
def vm_impl_flatten_grad(self):
    """Generate vm_impl function for FlattenGrad"""

    def vm_impl(dout, x):
        dout = dout.asnumpy()
        dout = vm.flatten_grad(dout, x)
        return Tensor(dout)

    return vm_impl


@vm_impl_getters.register(P.BiasAdd)
def vm_impl_bias_add(self):
    """Generate vm_impl function for BiasAdd"""

    def vm_impl(wx, bias):
        wx = wx.asnumpy()
        bias = bias.asnumpy()
        out = wx + bias
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(G.BiasAddGrad)
def vm_impl_bias_add_grad(self):
    """Generate vm_impl function for BiasAddGrad"""

    def vm_impl(dout):
        dout = dout.asnumpy()
        shape = np.shape(dout)
        return Tensor(np.add.reduce(dout, axis=tuple(range(len(shape) - 1))))

    return vm_impl


@vm_impl_getters.register(P.SoftmaxCrossEntropyWithLogits)
def vm_impl_softmax_cross_entropy_with_logits(self):
    """Generate vm_impl function for SoftmaxCrossEntropyWithLogits"""

    def vm_impl(logits, labels):
        logits = logits.asnumpy()
        labels = labels.asnumpy()
        loss, dx = vm.softmax_cross_entropy_with_logits(logits, labels)
        return (Tensor(np.array(loss)), Tensor(dx))

    return vm_impl


@vm_impl_getters.register(P.SparseSoftmaxCrossEntropyWithLogits)
def vm_impl_sparse_softmax_cross_entropy_with_logits(self):
    """Generate vm_impl function for SparseSoftmaxCrossEntropyWithLogits"""

    def vm_impl(logits, labels):
        logits = logits.asnumpy()
        labels = labels.asnumpy()

        n_class = labels.max() + 1
        n_sample = labels.shape[0]
        one_hot_label = np.zeros((n_sample, n_class))  # 3个样本，4个类别
        one_hot_label[:, labels] = 1  # 非零列赋值为1
        loss, dx = vm.softmax_cross_entropy_with_logits(logits, one_hot_label)
        if self.is_grad:
            return (Tensor(dx),)
        return (Tensor(np.array(loss)),)

    return vm_impl


@vm_impl_getters.register(P.ApplyMomentum)
def vm_impl_momentum(self):
    """Generate vm_impl function for Momentum"""

    def vm_impl(variable,
                accumulation,
                learning_rate,
                gradient,
                momentum,
                use_nesterov=False):
        gradient = gradient.asnumpy()
        accumulation = accumulation.asnumpy()
        variable = variable.asnumpy()
        shape = accumulation.shape
        learning_rate = np.full(shape, learning_rate.asnumpy())
        momentum = np.full(shape, momentum.asnumpy())
        accumulation = accumulation * momentum + gradient
        if use_nesterov is True:
            variable -= gradient * learning_rate + accumulation * momentum * learning_rate
        else:
            variable -= accumulation * learning_rate
        return Tensor(variable)

    return vm_impl


@vm_impl_getters.register(P.ResizeBilinear)
def vm_impl_resize_bilinear(self):
    """Generate vm_impl function for ResizeBilinear"""

    def vm_impl(x):
        out = vm.ResizeBilinear(x)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(G.ResizeBilinearGrad)
def vm_impl_resize_bilinear_grad(self):
    """Generate vm_impl function for ResizeBilinearGrad"""

    def vm_impl(dout, original_image):
        out = vm.ResizeBilinearGrad(dout, original_image)
        return Tensor(out)

    return vm_impl
