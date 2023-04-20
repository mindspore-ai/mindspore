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
"""Convolutional variational layers."""
from mindspore.ops import operations as P
from mindspore._checkparam import twice
from ...layer.conv import _Conv
from .layer_distribution import NormalPrior, normal_post_fn
from ._util import check_prior, check_posterior

__all__ = ['ConvReparam']


class _ConvVariational(_Conv):
    """
    Base class for all convolutional variational layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_prior_fn=NormalPrior,
                 weight_posterior_fn=normal_post_fn,
                 bias_prior_fn=NormalPrior,
                 bias_posterior_fn=normal_post_fn):
        kernel_size = twice(kernel_size)
        stride = twice(stride)
        dilation = twice(dilation)
        super(_ConvVariational, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init='normal',
            bias_init='zeros'
        )
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError('Attr \'pad_mode\' of \'Conv2d\' Op passed '
                             + str(pad_mode) + ', should be one of values in \'valid\', \'same\', \'pad\'.')

        # convolution args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.has_bias = has_bias

        self.shape = [self.out_channels, self.in_channels // self.group, *self.kernel_size]

        self.weight.requires_grad = False
        self.weight_prior = check_prior(weight_prior_fn, "weight_prior_fn")
        self.weight_posterior = check_posterior(weight_posterior_fn, shape=self.shape, param_name='bnn_weight',
                                                arg_name="weight_posterior_fn")

        if self.has_bias:
            self.bias.requires_grad = False
            self.bias_prior = check_prior(bias_prior_fn, "bias_prior_fn")
            self.bias_posterior = check_posterior(bias_posterior_fn, shape=[self.out_channels], param_name='bnn_bias',
                                                  arg_name="bias_posterior_fn")

        # mindspore operations
        self.bias_add = P.BiasAdd()
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group)

        self.log = P.Log()
        self.sum = P.ReduceSum()

    def construct(self, inputs):
        outputs = self._apply_variational_weight(inputs)
        if self.has_bias:
            outputs = self.apply_variational_bias(outputs)
        return outputs

    def extend_repr(self):
        """Display instance object as string."""
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, pad_mode={}, ' \
            'padding={}, dilation={}, group={}, weight_mean={}, weight_std={}, has_bias={}' \
            .format(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.pad_mode, self.padding,
                    self.dilation, self.group, self.weight_posterior.mean, self.weight_posterior.untransformed_std,
                    self.has_bias)
        if self.has_bias:
            s += ', bias_mean={}, bias_std={}' \
                .format(self.bias_posterior.mean, self.bias_posterior.untransformed_std)
        return s

    def compute_kl_loss(self):
        """Compute kl loss"""
        weight_type = self.weight_posterior("get_dist_type")
        weight_args_list = self.weight_posterior("get_dist_args")

        kl = self.weight_prior("kl_loss", weight_type, *weight_args_list)
        kl_loss = self.sum(kl)
        if self.has_bias:
            bias_args_list = self.bias_posterior("get_dist_args")
            bias_type = self.bias_posterior("get_dist_type")

            kl = self.bias_prior("kl_loss", bias_type, *bias_args_list)
            kl = self.sum(kl)
            kl_loss += kl
        return kl_loss

    def apply_variational_bias(self, inputs):
        """Calculate bias."""
        bias_posterior_tensor = self.bias_posterior("sample")
        return self.bias_add(inputs, bias_posterior_tensor)


class ConvReparam(_ConvVariational):
    r"""
    Convolutional variational layers with Reparameterization.

    For more details, refer to the paper `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The data type is an integer or
            a tuple of 2 integers. The kernel size specifies the height and
            width of the 2D convolution window. a single integer stands for the
            value is for both height and width of the kernel. With the `kernel_size`
            being a tuple of 2 integers, the first value is for the height and the other
            is the width of the kernel.
        stride(Union[int, tuple[int]]): The distance of kernel moving,
            an integer number represents that the height and width of movement
            are both strides, or a tuple of two integers numbers represents that
            height and width of movement respectively. Default: ``1`` .
        pad_mode (str): Specifies the padding mode. The optional values are
            ``"same"`` , ``"valid"`` , and ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Adopts the way of completion. Output height and width
              will be the same as the input.
              The total number of padding will be calculated for in horizontal and
              vertical directions and evenly distributed to top and bottom,
              left and right if possible. Otherwise, the last extra padding
              will be done from the bottom and the right side. If this mode
              is set, `padding` must be 0.

            - ``"valid"``: Adopts the way of discarding. The possible largest
              height and width of the output will be returned without padding.
              Extra pixels will be discarded. If this mode is set, `padding`
              must be 0.

            - ``"pad"``: Implicit paddings on both sides of the input. The number
              of `padding` will be padded to the input Tensor borders.
              `padding` must be greater than or equal to 0.

        padding (Union[int, tuple[int]]): Implicit paddings on both sides of
            the input. Default: ``0`` .
        dilation (Union[int, tuple[int]]): The data type is an integer or a tuple
            of 2 integers. This parameter specifies the dilation rate of the
            dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling
            location. Its value must be greater or equal to 1 and bounded
            by the height and width of the input. Default: ``1`` .
        group (int): Splits filter into groups, `in_ channels` and
            `out_channels` must be divisible by the number of groups.
            Default: ``1`` .
        has_bias (bool): Specifies whether the layer uses a bias vector.
            Default: ``False`` .
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
        - **input** (Tensor) - The shape of the tensor is :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, with the shape being :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.nn.probability import bnn_layers
        >>> net = bnn_layers.ConvReparam(120, 240, 4, has_bias=False)
        >>> input = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> output = net(input).shape
        >>> print(output)
        (1, 240, 1024, 640)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            pad_mode='same',
            padding=0,
            dilation=1,
            group=1,
            has_bias=False,
            weight_prior_fn=NormalPrior,
            weight_posterior_fn=normal_post_fn,
            bias_prior_fn=NormalPrior,
            bias_posterior_fn=normal_post_fn):
        super(ConvReparam, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            padding=padding,
            dilation=dilation,
            group=group,
            has_bias=has_bias,
            weight_prior_fn=weight_prior_fn,
            weight_posterior_fn=weight_posterior_fn,
            bias_prior_fn=bias_prior_fn,
            bias_posterior_fn=bias_posterior_fn
        )

    def _apply_variational_weight(self, inputs):
        """Calculate weight."""
        weight_posterior_tensor = self.weight_posterior("sample")
        outputs = self.conv2d(inputs, weight_posterior_tensor)
        return outputs
