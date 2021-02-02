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
"""
loss function for training and sample function for testing
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import context


class log_sum_exp(nn.Cell):
    """Numerically stable log_sum_exp
    """

    def __init__(self):
        super(log_sum_exp, self).__init__()
        self.maxi = P.ReduceMax()
        self.maxi_dim = P.ReduceMax(keep_dims=True)
        self.log = P.Log()
        self.sums = P.ReduceSum()
        self.exp = P.Exp()

    def construct(self, x):
        axis = len(x.shape) - 1
        m = self.maxi(x, axis)
        m2 = self.maxi_dim(x, axis)
        return m + self.log(self.sums(self.exp(x - m2), axis))


class log_softmax(nn.Cell):
    """
    replacement of P.LogSoftmax(-1) in CPU mode
        only support x.shape == 2 or 3
    """

    def __init__(self):
        super(log_softmax, self).__init__()
        self.maxi = P.ReduceMax()
        self.log = P.Log()
        self.sums = P.ReduceSum()
        self.exp = P.Exp()
        self.axis = -1
        self.concat = P.Concat(-1)
        self.expanddims = P.ExpandDims()

    def construct(self, x):
        """

        Args:
            x (Tensor): input

        Returns:
            Tensor: log_softmax of input

        """
        c = self.maxi(x, self.axis)
        logs, lsm = None, None
        if len(x.shape) == 2:
            for j in range(x.shape[-1]):
                temp = self.expanddims(self.exp(x[:, j] - c), -1)
                logs = temp if j == 0 else self.concat((logs, temp))
            sums = self.sums(logs, -1)
            for i in range(x.shape[-1]):
                temp = self.expanddims(x[:, i] - c - self.log(sums), -1)
                lsm = temp if i == 0 else self.concat((lsm, temp))
            return lsm
        if len(x.shape) == 3:
            for j in range(x.shape[-1]):
                temp = self.expanddims(self.exp(x[:, :, j] - c), -1)
                logs = temp if j == 0 else self.concat((logs, temp))
            sums = self.sums(logs, -1)
            for i in range(x.shape[-1]):
                temp = self.expanddims(x[:, :, i] - c - self.log(sums), -1)
                lsm = temp if i == 0 else self.concat((lsm, temp))
            return lsm
        return None


class Stable_softplus(nn.Cell):
    """Numerically stable softplus
    """

    def __init__(self):
        super(Stable_softplus, self).__init__()
        self.log_op = P.Log()
        self.abs_op = P.Abs()
        self.relu_op = P.ReLU()
        self.exp_op = P.Exp()

    def construct(self, x):
        return self.log_op(1 + self.exp_op(- self.abs_op(x))) + self.relu_op(x)


class discretized_mix_logistic_loss(nn.Cell):
    """
    Discretized_mix_logistic_loss

    Args:
        num_classes (int): Num_classes
        log_scale_min (float): Log scale minimum value

    """

    def __init__(self, num_classes=256, log_scale_min=-7.0, reduce=True):
        super(discretized_mix_logistic_loss, self).__init__()
        self.num_classes = num_classes
        self.log_scale_min = log_scale_min
        self.reduce = reduce
        self.transpose_op = P.Transpose()
        self.exp = P.Exp()
        self.sigmoid = P.Sigmoid()
        self.softplus = Stable_softplus()
        self.log = P.Log()
        self.cast = P.Cast()
        self.expand_dims = P.ExpandDims()
        self.tile = P.Tile()
        self.maximum = P.Maximum()
        self.sums = P.ReduceSum()
        self.lse = log_sum_exp()
        self.reshape = P.Reshape()
        self.factor = self.log(Tensor((self.num_classes - 1) / 2, ms.float32))
        self.tensor_one = Tensor(1., ms.float32)

        if context.get_context("device_target") == "CPU":
            self.logsoftmax = log_softmax()
        else:
            self.logsoftmax = P.LogSoftmax(-1)

    def construct(self, y_hat, y):
        """

        Args:
            y_hat (Tensor): Predicted distribution
            y (Tensor): Target

        Returns:
            Tensor: Discretized_mix_logistic_loss

        """
        nr_mix = y_hat.shape[1] // 3

        # (B x T x C)
        y_hat = self.transpose_op(y_hat, (0, 2, 1))

        # (B, T, num_mixtures) x 3
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix:2 * nr_mix]
        min_cut = self.log_scale_min * self.tile(self.tensor_one, (y_hat.shape[0], y_hat.shape[1], nr_mix))
        log_scales = self.maximum(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min_cut)

        # B x T x 1 -> B x T x num_mixtures
        y = self.tile(y, (1, 1, nr_mix))

        centered_y = y - means
        inv_stdv = self.exp(-log_scales)
        plus_in = inv_stdv * (centered_y + 1. / (self.num_classes - 1))
        cdf_plus = self.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1. / (self.num_classes - 1))
        cdf_min = self.sigmoid(min_in)

        log_cdf_plus = plus_in - self.softplus(plus_in)

        log_one_minus_cdf_min = -self.softplus(min_in)

        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered_y
        log_pdf_mid = mid_in - log_scales - 2. * self.softplus(mid_in)

        inner_inner_cond = self.cast(cdf_delta > 1e-5, ms.float32)
        min_cut2 = 1e-12 * self.tile(self.tensor_one, cdf_delta.shape)
        inner_inner_out = inner_inner_cond * \
                          self.log(self.maximum(cdf_delta, min_cut2)) + \
                          (1. - inner_inner_cond) * (log_pdf_mid - self.factor)
        inner_cond = self.cast(y > 0.999, ms.float32)
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = self.cast(y < -0.999, ms.float32)
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

        a, b, c = logit_probs.shape[0], logit_probs.shape[1], logit_probs.shape[2]
        logit_probs = self.logsoftmax(self.reshape(logit_probs, (-1, c)))
        logit_probs = self.reshape(logit_probs, (a, b, c))

        log_probs = log_probs + logit_probs
        if self.reduce:
            return -self.sums(self.lse(log_probs))
        return self.expand_dims(-self.lse(log_probs), -1)


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (ndarray): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        ndarray
    """
    nr_mix = y.shape[1] // 3

    # B x T x C
    y = np.transpose(y, (0, 2, 1))
    logit_probs = y[:, :, :nr_mix]

    temp = np.random.uniform(1e-5, 1.0 - 1e-5, logit_probs.shape)
    temp = logit_probs - np.log(- np.log(temp))

    argmax = np.argmax(temp, axis=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = np.eye(nr_mix)[argmax]
    means = np.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, axis=-1)
    log_scales = np.clip(np.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, axis=-1), a_min=log_scale_min, a_max=None)

    u = np.random.uniform(1e-5, 1.0 - 1e-5, means.shape)
    x = means + np.exp(log_scales) * (np.log(u) - np.log(1. - u))
    x = np.clip(x, -1., 1.)
    return x.astype(np.float32)


class mix_gaussian_loss(nn.Cell):
    """
    Mix gaussian loss
    """

    def __init__(self, log_scale_min=-7.0, reduce=True):
        super(mix_gaussian_loss, self).__init__()
        self.log_scale_min = log_scale_min
        self.reduce = reduce
        self.transpose_op = P.Transpose()
        self.maximum = P.Maximum()
        self.tile = P.Tile()
        self.exp = P.Exp()
        self.expand_dims = P.ExpandDims()
        self.sums = P.ReduceSum()
        self.lse = log_sum_exp()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()
        self.const = P.ScalarToArray()
        self.log = P.Log()
        self.tensor_one = Tensor(1., ms.float32)

        if context.get_context("device_target") == "CPU":
            self.logsoftmax = log_softmax()
        else:
            self.logsoftmax = P.LogSoftmax(-1)

    def construct(self, y_hat, y):
        """

        Args:
            y_hat (Tensor): Predicted probability
            y (Tensor): Target

        Returns:
            Tensor: Mix_gaussian_loss

        """
        C = y_hat.shape[1]
        if C == 2:
            nr_mix = 1
        else:
            nr_mix = y_hat.shape[1] // 3

        # (B x T x C)
        y_hat = self.transpose_op(y_hat, (0, 2, 1))

        if C == 2:
            logit_probs = None
            means = y_hat[:, :, 0:1]
            min_cut = self.log_scale_min * self.tile(self.tensor_one, (y_hat.shape[0], y_hat.shape[1], 1))
            log_scales = self.maximum(y_hat[:, :, 1:2], min_cut)
        else:
            # (B, T, num_mixtures) x 3
            logit_probs = y_hat[:, :, :nr_mix]
            means = y_hat[:, :, nr_mix:2 * nr_mix]
            min_cut = self.log_scale_min * self.tile(self.tensor_one, (y_hat.shape[0], y_hat.shape[1], nr_mix))
            log_scales = self.maximum(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min_cut)

        # B x T x 1 -> B x T x num_mixtures
        y = self.tile(y, (1, 1, nr_mix))
        centered_y = y - means

        sd = self.exp(log_scales)
        unnormalized_log_prob = -1. * (self.sq(centered_y - 0.)) / (2. * self.sq(sd))
        neg_normalization = -1. * self.log(self.const(2. * np.pi)) / 2. - self.log(sd)
        log_probs = unnormalized_log_prob + neg_normalization

        if nr_mix > 1:
            log_probs = log_probs + self.logsoftmax(logit_probs)

        if self.reduce:
            if nr_mix == 1:
                return -self.sums(log_probs)
            return -self.sums(self.lse(log_probs))
        if nr_mix == 1:
            return -log_probs
        return self.expand_dims(-self.lse(log_probs), -1)


def sample_from_mix_gaussian(y, log_scale_min=-7.0):
    """
    Sample_from_mix_gaussian

    Args:
        y (ndarray): B x C x T

    Returns:
        ndarray

    """
    C = y.shape[1]
    if C == 2:
        nr_mix = 1
    else:
        nr_mix = y.shape[1] // 3

    # B x T x C
    y = np.transpose(y, (0, 2, 1))

    if C == 2:
        logit_probs = None
    else:
        logit_probs = y[:, :, :nr_mix]

    if nr_mix > 1:
        temp = np.random.uniform(1e-5, 1.0 - 1e-5, logit_probs.shape)
        temp = logit_probs - np.log(- np.log(temp))
        argmax = np.argmax(temp, axis=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = np.eye(nr_mix)[argmax]

        means = np.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, axis=-1)

        log_scales = np.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, axis=-1)
    else:
        if C == 2:
            means, log_scales = y[:, :, 0], y[:, :, 1]
        elif C == 3:
            means, log_scales = y[:, :, 1], y[:, :, 2]
        else:
            assert False, "shouldn't happen"

    scales = np.exp(log_scales)
    x = np.random.normal(loc=means, scale=scales)
    x = np.clip(x, -1., 1.)
    return x.astype(np.float32)


# self-implemented onehotcategorical distribution
# https://zhuanlan.zhihu.com/p/59550457
def sample_from_mix_onehotcategorical(x):
    """
    Sample_from_mix_onehotcategorical

    Args:
        x (ndarray): Predicted softmax probability

    Returns:
        ndarray

    """
    pi = np.log(x)
    u = np.random.uniform(0, 1, x.shape)
    g = -np.log(-np.log(u))
    c = np.argmax(pi + g, axis=1)
    return np.array(np.eye(256)[c], dtype=np.float32)
