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
from mindspore import _checkparam as Validator
from ...cell import Cell
from ...layer.activation import get_activation
from ..distribution.normal import Normal
from .layer_distribution import NormalPrior, normal_post_fn
from ._util import check_prior, check_posterior

__all__ = ['DenseReparam', 'DenseLocalReparam']


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
            weight_posterior_fn=normal_post_fn,
            bias_prior_fn=NormalPrior,
            bias_posterior_fn=normal_post_fn):
        super(_DenseVariational, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)

        self.weight_prior = check_prior(weight_prior_fn, "weight_prior_fn")
        self.weight_posterior = check_posterior(weight_posterior_fn, shape=[self.out_channels, self.in_channels],
                                                param_name='bnn_weight', arg_name="weight_posterior_fn")

        if self.has_bias:
            self.bias_prior = check_prior(bias_prior_fn, "bias_prior_fn")
            self.bias_posterior = check_posterior(bias_posterior_fn, shape=[self.out_channels], param_name='bnn_bias',
                                                  arg_name="bias_posterior_fn")

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
            outputs = self.apply_variational_bias(outputs)
        if self.activation_flag:
            outputs = self.activation(outputs)
        return outputs

    def extend_repr(self):
        """Display instance object as string."""
        s = 'in_channels={}, out_channels={}, weight_mean={}, weight_std={}, has_bias={}' \
            .format(self.in_channels, self.out_channels, self.weight_posterior.mean,
                    self.weight_posterior.untransformed_std, self.has_bias)
        if self.has_bias:
            s += ', bias_mean={}, bias_std={}' \
                .format(self.bias_posterior.mean, self.bias_posterior.untransformed_std)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s

    def apply_variational_bias(self, inputs):
        """Calculate bias."""
        bias_posterior_tensor = self.bias_posterior("sample")
        return self.bias_add(inputs, bias_posterior_tensor)

    def compute_kl_loss(self):
        """Compute kl loss"""
        weight_args_list = self.weight_posterior("get_dist_args")
        weight_type = self.weight_posterior("get_dist_type")

        kl = self.weight_prior("kl_loss", weight_type, *weight_args_list)
        kl_loss = self.sum(kl)
        if self.has_bias:
            bias_args_list = self.bias_posterior("get_dist_args")
            bias_type = self.bias_posterior("get_dist_type")

            kl = self.bias_prior("kl_loss", bias_type, *bias_args_list)
            kl = self.sum(kl)
            kl_loss += kl
        return kl_loss


class DenseReparam(_DenseVariational):
    r"""
    Dense variational layers with Reparameterization.

    For more details, refer to the paper `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_.

    Applies dense-connected layer to the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{weight} + \text{bias}),

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
        activation (str, Cell): A regularization function applied to the output of the layer.
            The type of `activation` can be a string (eg. 'relu') or a Cell (eg. nn.ReLU()).
            Note that if the type of activation is Cell, it must be instantiated beforehand.
            Default: ``None`` .
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``False`` .
        weight_prior_fn (Cell): The prior distribution for weight.
            It must return a mindspore distribution instance.
            Default: ``NormalPrior`` . (which creates an instance of standard
            normal distribution). The current version only supports normal distribution.
        weight_posterior_fn (function): The posterior distribution for sampling weight.
            It must be a function handle which returns a mindspore
            distribution instance. Default: ``normal_post_fn`` .
            The current version only supports normal distribution.
        bias_prior_fn (Cell): The prior distribution for bias vector. It must return
            a mindspore distribution. Default: ``NormalPrior`` (which creates an
            instance of standard normal distribution). The current version
            only supports normal distribution.
        bias_posterior_fn (function): The posterior distribution for sampling bias vector.
            It must be a function handle which returns a mindspore
            distribution instance. Default: ``normal_post_fn`` .
            The current version only supports normal distribution.

    Inputs:
        - **input** (Tensor) - The shape of the tensor is :math:`(N, in\_channels)`.

    Outputs:
        Tensor, the shape of the tensor is :math:`(N, out\_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.nn.probability import bnn_layers
        >>> net = bnn_layers.DenseReparam(3, 4)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> output = net(input).shape
        >>> print(output)
        (2, 4)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            activation=None,
            has_bias=True,
            weight_prior_fn=NormalPrior,
            weight_posterior_fn=normal_post_fn,
            bias_prior_fn=NormalPrior,
            bias_posterior_fn=normal_post_fn):
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
        """Calculate weight."""
        weight_posterior_tensor = self.weight_posterior("sample")
        outputs = self.matmul(inputs, weight_posterior_tensor)
        return outputs


class DenseLocalReparam(_DenseVariational):
    r"""
    Dense variational layers with Local Reparameterization.

    For more details, refer to the paper `Variational Dropout and the Local Reparameterization
    Trick <https://arxiv.org/abs/1506.02557>`_.

    Applies dense-connected layer to the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{weight} + \text{bias}),

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
        activation (str, Cell): A regularization function applied to the output of the layer.
            The type of `activation` can be a string (eg. 'relu') or a Cell (eg. nn.ReLU()).
            Note that if the type of activation is Cell, it must be instantiated beforehand.
            Default: ``None`` .
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``False`` .
        weight_prior_fn (Cell): The prior distribution for weight.
            It must return a mindspore distribution instance.
            Default: ``NormalPrior`` . (which creates an instance of standard
            normal distribution). The current version only supports normal distribution.
        weight_posterior_fn (function): The posterior distribution for sampling weight.
            It must be a function handle which returns a mindspore
            distribution instance. Default: ``normal_post_fn`` .
            The current version only supports normal distribution.
        bias_prior_fn (Cell): The prior distribution for bias vector. It must return
            a mindspore distribution. Default: ``NormalPrior`` (which creates an
            instance of standard normal distribution). The current version
            only supports normal distribution.
        bias_posterior_fn (function): The posterior distribution for sampling bias vector.
            It must be a function handle which returns a mindspore
            distribution instance. Default: ``normal_post_fn`` .
            The current version only supports normal distribution.

    Inputs:
        - **input** (Tensor) - The shape of the tensor is :math:`(N, in\_channels)`.

    Outputs:
        Tensor, the shape of the tensor is :math:`(N, out\_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.nn.probability import bnn_layers
        >>> net = bnn_layers.DenseLocalReparam(3, 4)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> output = net(input).shape
        >>> print(output)
        (2, 4)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            activation=None,
            has_bias=True,
            weight_prior_fn=NormalPrior,
            weight_posterior_fn=normal_post_fn,
            bias_prior_fn=NormalPrior,
            bias_posterior_fn=normal_post_fn):
        super(DenseLocalReparam, self).__init__(
            in_channels,
            out_channels,
            activation=activation,
            has_bias=has_bias,
            weight_prior_fn=weight_prior_fn,
            weight_posterior_fn=weight_posterior_fn,
            bias_prior_fn=bias_prior_fn,
            bias_posterior_fn=bias_posterior_fn
        )
        self.sqrt = P.Sqrt()
        self.square = P.Square()
        self.normal = Normal()

    def _apply_variational_weight(self, inputs):
        """Calculate weight."""
        mean = self.matmul(inputs, self.weight_posterior("mean"))
        std = self.sqrt(self.matmul(self.square(inputs), self.square(self.weight_posterior("sd"))))
        weight_posterior_affine_tensor = self.normal("sample", mean=mean, sd=std)
        return weight_posterior_affine_tensor
