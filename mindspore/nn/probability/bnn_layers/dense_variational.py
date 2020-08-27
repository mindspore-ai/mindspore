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
"""dense_variational"""
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore._checkparam import check_int_positive, check_bool
from ...cell import Cell
from ...layer.activation import get_activation
from .layer_distribution import NormalPrior, NormalPosterior

__all__ = ['DenseReparam']


class _DenseVariational(Cell):
    """
    Base class for all dense variational layers.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            activation=None,
            has_bias=True,
            weight_prior_fn=NormalPrior,
            weight_posterior_fn=lambda name, shape: NormalPosterior(name=name, shape=shape),
            bias_prior_fn=NormalPrior,
            bias_posterior_fn=lambda name, shape: NormalPosterior(name=name, shape=shape)):
        super(_DenseVariational, self).__init__()
        self.in_channels = check_int_positive(in_channels)
        self.out_channels = check_int_positive(out_channels)
        self.has_bias = check_bool(has_bias)

        if isinstance(weight_prior_fn, Cell):
            self.weight_prior = weight_prior_fn
        else:
            self.weight_prior = weight_prior_fn()
        for prior_name, prior_dist in self.weight_prior.name_cells().items():
            if prior_name != 'normal':
                raise TypeError("The type of distribution of `weight_prior_fn` should be `normal`")
            if not (isinstance(getattr(prior_dist, '_mean_value'), Tensor) and
                    isinstance(getattr(prior_dist, '_sd_value'), Tensor)):
                raise TypeError("The input form of `weight_prior_fn` is incorrect")

        try:
            self.weight_posterior = weight_posterior_fn(shape=[self.out_channels, self.in_channels], name='bnn_weight')
        except TypeError:
            raise TypeError('The type of `weight_posterior_fn` should be `NormalPosterior`')
        for posterior_name, _ in self.weight_posterior.name_cells().items():
            if posterior_name != 'normal':
                raise TypeError("The type of distribution of `weight_posterior_fn` should be `normal`")

        if self.has_bias:
            if isinstance(bias_prior_fn, Cell):
                self.bias_prior = bias_prior_fn
            else:
                self.bias_prior = bias_prior_fn()
            for prior_name, prior_dist in self.bias_prior.name_cells().items():
                if prior_name != 'normal':
                    raise TypeError("The type of distribution of `bias_prior_fn` should be `normal`")
                if not (isinstance(getattr(prior_dist, '_mean_value'), Tensor) and
                        isinstance(getattr(prior_dist, '_sd_value'), Tensor)):
                    raise TypeError("The input form of `bias_prior_fn` is incorrect")

            try:
                self.bias_posterior = bias_posterior_fn(shape=[self.out_channels], name='bnn_bias')
            except TypeError:
                raise TypeError('The type of `bias_posterior_fn` should be `NormalPosterior`')
            for posterior_name, _ in self.bias_posterior.name_cells().items():
                if posterior_name != 'normal':
                    raise TypeError("The type of distribution of `bias_posterior_fn` should be `normal`")

        self.activation = activation
        if not self.activation:
            self.activation_flag = False
        else:
            self.activation_flag = True
            if isinstance(self.activation, str):
                self.activation = get_activation(activation)
            elif isinstance(self.activation, Cell):
                self.activation = activation
            else:
                raise ValueError('The type of `activation` is wrong.')

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()
        self.sum = P.ReduceSum()

    def construct(self, x):
        outputs = self._apply_variational_weight(x)
        if self.has_bias:
            outputs = self._apply_variational_bias(outputs)
        if self.activation_flag:
            outputs = self.activation(outputs)
        return outputs

    def extend_repr(self):
        str_info = 'in_channels={}, out_channels={}, weight_mean={}, weight_std={}, has_bias={}' \
            .format(self.in_channels, self.out_channels, self.weight_posterior.mean,
                    self.weight_posterior.untransformed_std, self.has_bias)
        if self.has_bias:
            str_info = str_info + ', bias_mean={}, bias_std={}' \
                .format(self.bias_posterior.mean, self.bias_posterior.untransformed_std)

        if self.activation_flag:
            str_info = str_info + ', activation={}'.format(self.activation)
        return str_info

    def _apply_variational_bias(self, inputs):
        bias_posterior_tensor = self.bias_posterior("sample")
        return self.bias_add(inputs, bias_posterior_tensor)

    def compute_kl_loss(self):
        """Compute kl loss."""
        weight_post_mean = self.weight_posterior("mean")
        weight_post_sd = self.weight_posterior("sd")

        kl = self.weight_prior("kl_loss", "Normal", weight_post_mean, weight_post_sd)
        kl_loss = self.sum(kl)
        if self.has_bias:
            bias_post_mean = self.bias_posterior("mean")
            bias_post_sd = self.bias_posterior("sd")

            kl = self.bias_prior("kl_loss", "Normal", bias_post_mean, bias_post_sd)
            kl = self.sum(kl)
            kl_loss += kl
        return kl_loss


class DenseReparam(_DenseVariational):
    r"""
    Dense variational layers with Reparameterization.

    See more details in paper `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_.

    Applies dense-connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{kernel} + \text{bias}),

    where :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{activation}` is a weight matrix with the same
    data type as the inputs created by the layer, :math:`\text{weight}` is a weight
    matrix sampling from posterior distribution of weight, and :math:`\text{bias}` is a
    bias vector with the same data type as the inputs created by the layer (only if
    has_bias is True). The bias vector is sampling from posterior distribution of
    :math:`\text{bias}`.

    Args:
        in_channels (int): The number of input channel.
        out_channels (int): The number of output channel .
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        activation (str, Cell): Regularizer function applied to the output of the layer. The type of activation can
            be str (eg. 'relu') or Cell (eg. nn.ReLU()). Note that if the type of activation is Cell, it must have been
            instantiated. Default: None.
        weight_prior_fn: prior distribution for weight.
            It should return a mindspore distribution instance.
            Default: NormalPrior. (which creates an instance of standard
            normal distribution). The current version only supports normal distribution.
        weight_posterior_fn: posterior distribution for sampling weight.
            It should be a function handle which returns a mindspore
            distribution instance. Default: lambda name, shape: NormalPosterior(name=name, shape=shape).
            The current version only supports normal distribution.
        bias_prior_fn: prior distribution for bias vector. It should return
            a mindspore distribution. Default: NormalPrior(which creates an
            instance of standard normal distribution). The current version
            only supports normal distribution.
        bias_posterior_fn: posterior distribution for sampling bias vector.
            It should be a function handle which returns a mindspore
            distribution instance. Default: lambda name, shape: NormalPosterior(name=name, shape=shape).
            The current version only supports normal distribution.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`.

    Examples:
        >>> net = DenseReparam(3, 4)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> net(input).shape
        (2, 4)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            activation=None,
            has_bias=True,
            weight_prior_fn=NormalPrior,
            weight_posterior_fn=lambda name, shape: NormalPosterior(name=name, shape=shape),
            bias_prior_fn=NormalPrior,
            bias_posterior_fn=lambda name, shape: NormalPosterior(name=name, shape=shape)):
        super(DenseReparam, self).__init__(
            in_channels,
            out_channels,
            activation=activation,
            has_bias=has_bias,
            weight_prior_fn=weight_prior_fn,
            weight_posterior_fn=weight_posterior_fn,
            bias_prior_fn=bias_prior_fn,
            bias_posterior_fn=bias_posterior_fn
        )

    def _apply_variational_weight(self, inputs):
        weight_posterior_tensor = self.weight_posterior("sample")
        outputs = self.matmul(inputs, weight_posterior_tensor)
        return outputs
