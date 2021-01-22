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
"""normalization"""
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops.primitive import constexpr
import mindspore.context as context
from mindspore._checkparam import Validator as validator
from mindspore._extends import cell_attr_register
from mindspore.communication.management import get_group_size, get_rank
from mindspore.communication import management
from mindspore.ops import _selected_ops
from ..cell import Cell

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'GroupNorm', 'GlobalBatchNorm']


class _BatchNorm(Cell):
    """Batch Normalization base class."""

    @cell_attr_register
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros',
                 moving_mean_init='zeros',
                 moving_var_init='ones',
                 use_batch_statistics=None,
                 device_num_each_group=1,
                 input_dims='2d',
                 data_format='NCHW'):
        super(_BatchNorm, self).__init__()
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if momentum < 0 or momentum > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(momentum))
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.use_batch_statistics = use_batch_statistics
        self.num_features = num_features
        self.eps = eps
        self.input_dims = input_dims
        self.moving_mean = Parameter(initializer(
            moving_mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer(
            moving_var_init, num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)
        self.group = validator.check_positive_int(device_num_each_group)
        self.is_global = False
        if self.group != 1:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            self.device_list = [i for i in range(0, self.rank_size)]
            self.rank_list = self.list_group(self.device_list, self.group)
            self.rank_list_idx = len(self.rank_list)
            for i in range(self.rank_list_idx):
                if self.rank_id in self.rank_list[i] and self.group != 1:
                    self.is_global = True
                    management.create_group('group' + str(i), self.rank_list[i])
                    self.all_reduce = P.AllReduce(P.ReduceOp.SUM, 'group' + str(i)).add_prim_attr('fusion', 1)
        self.shape = P.Shape()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.reshape = P.Reshape()
        self._target = context.get_context("device_target")
        self.is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        self.momentum = 1.0 - momentum
        if context.get_context("enable_ge"):
            self.is_ge_backend = True
        else:
            self.is_ge_backend = False

        if self._target == "Ascend":
            self.bn_train = P.BatchNorm(is_training=True,
                                        epsilon=self.eps,
                                        momentum=self.momentum)
        if self._target == "GPU":
            self.bn_train = P.FusedBatchNormEx(mode=1,
                                               epsilon=self.eps,
                                               momentum=self.momentum,
                                               data_format=self.format)
        if self._target == "CPU":
            self.bn_train = P.FusedBatchNorm(mode=1,
                                             epsilon=self.eps,
                                             momentum=self.momentum)
        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format=self.format)
        self.enable_global_sync = self.is_global and (self.is_ge_backend or\
            (self.is_graph_mode and self._target == "Ascend"))

        data_parallel_strategy = ((1,), (1,))
        data_parallel_strategy_one = ((1,), ())
        self.sub_mean = P.Sub().shard(data_parallel_strategy)
        self.sub_var = P.Sub().shard(data_parallel_strategy)
        self.mul_mean = P.Mul().shard(data_parallel_strategy_one)
        self.mul_var = P.Mul().shard(data_parallel_strategy_one)
        self.assign_sub_mean = P.AssignSub().shard(data_parallel_strategy)
        self.assign_sub_var = P.AssignSub().shard(data_parallel_strategy)

    def _check_data_dim(self, x):
        raise NotImplementedError

    def list_group(self, world_rank, group_size):
        if group_size > get_group_size():
            raise ValueError("group size can not be greater than local rank size, group size is {}, "
                             "local_rank_size is {}".format(group_size, get_group_size()))
        if len(world_rank) % group_size != 0:
            raise ValueError("please make your group size correct.")
        world_rank_list = zip(*(iter(world_rank),) * group_size)
        group_list = [list(i) for i in world_rank_list]
        return group_list

    def _global_sync(self, x, axes, re_shape):
        """calculate global batch normalization output"""
        x_mean = self.reduce_mean(x, axes)
        x_mean_square = self.reduce_mean(self.square(x), axes)
        global_batch_mean = self.all_reduce(x_mean) / self.group
        global_batch_mean_square = self.all_reduce(x_mean_square) / self.group
        global_mean = global_batch_mean
        global_var = global_batch_mean_square - self.square(global_mean)
        var_sqrt = self.sqrt(global_var + self.eps)
        mean_first = (x - global_mean) / var_sqrt
        y = mean_first * self.reshape(self.gamma, re_shape) + self.reshape(self.beta, re_shape)

        mean_sub = self.sub_mean(self.reshape(self.moving_mean, re_shape), global_mean)
        tmp_mean = self.mul_mean(mean_sub, self.cast(self.momentum, self.dtype(mean_sub)))
        mean_sub2 = self.sub_var(self.reshape(self.moving_mean, re_shape), global_var)
        tmp_variance = self.mul_var(mean_sub2, self.cast(self.momentum, self.dtype(mean_sub2)))
        y = F.depend(y, self.assign_sub_mean(self.moving_mean, self.reshape(tmp_mean, self.shape(self.moving_mean))))
        y = F.depend(y, self.assign_sub_var(self.moving_variance,
                                            self.reshape(tmp_variance, self.shape(self.moving_variance))))
        return y

    def construct(self, x):
        _shape_check_bn(self.shape(x), self.input_dims)
        if self.use_batch_statistics is None:
            flag = self.training
        else:
            flag = self.use_batch_statistics

        if flag:
            if self.enable_global_sync:
                axes, re_shape = _shape_infer(F.shape(x), self.num_features)
                return self._global_sync(x, axes, re_shape)

            return self.bn_train(x,
                                 self.gamma,
                                 self.beta,
                                 self.moving_mean,
                                 self.moving_variance)[0]

        return self.bn_infer(x,
                             self.gamma,
                             self.beta,
                             self.moving_mean,
                             self.moving_variance)[0]

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, gamma={}, beta={}, moving_mean={}, moving_variance={}'.format(
            self.num_features, self.eps, self.momentum, self.gamma, self.beta, self.moving_mean, self.moving_variance)


@constexpr
def _channel_check(channel, num_channel):
    if channel != num_channel:
        raise ValueError("the input channel is not equal with num_channel")


@constexpr
def _shape_check(in_shape):
    if len(in_shape) != 4:
        raise ValueError("The input must has 4 dims.")


@constexpr
def _shape_check_bn(in_shape, in_dims):
    dim = len(in_shape)
    if in_dims == '1d' and dim != 2:
        raise ValueError("The input must has 2 dims.")
    if in_dims == '2d' and dim != 4:
        raise ValueError("The input must has 4 dims.")
    if in_dims == 'both' and dim != 2 and dim != 4:
        raise ValueError("The input must has 2 dims or 4 dims.")


@constexpr
def _shape_infer(x_shape, num_feature):
    """global batch normalization shape and axes infer"""
    if len(x_shape) == 4:
        axes = (0, 2, 3)
        re_shape = (1, num_feature, 1, 1)
    else:
        axes = (0,)
        re_shape = (1, num_feature)
    return axes, re_shape


class BatchNorm1d(_BatchNorm):
    r"""
    Batch normalization layer over a 2D input.

    Batch Normalization is widely used in convolutional networks. This layer
    applies Batch Normalization over a 2D input (a mini-batch of 1D inputs) to
    reduce internal covariate shift as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It
    rescales and recenters the feature using a mini-batch of data and
    the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Note:
        The implementation of BatchNorm is different in graph mode and pynative mode, therefore the mode is not
        recommended to be changed after net was initialized.

    Args:
        num_features (int): `C` from an expected input of size (N, C).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data. If false,
            use the mean value and variance value of specified value. If None, the training process will use the mean
            and variance of current batch data and track the running mean and variance, the evaluation process will use
            the running mean and variance. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out})`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.BatchNorm1d(num_features=4)
        >>> np.random.seed(0)
        >>> input = Tensor(np.random.randint(0, 255, [2, 4]), mindspore.float32)
        >>> output = net(input)
        >>> print(output)
        [[171.99915   46.999763  116.99941  191.99904 ]
         [ 66.999664 250.99875   194.99902  102.99948 ]]
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros',
                 moving_mean_init='zeros',
                 moving_var_init='ones',
                 use_batch_statistics=None):
        super(BatchNorm1d, self).__init__(num_features,
                                          eps,
                                          momentum,
                                          affine,
                                          gamma_init,
                                          beta_init,
                                          moving_mean_init,
                                          moving_var_init,
                                          use_batch_statistics,
                                          input_dims='1d')

    def _check_data_dim(self, x):
        if x.ndim != 2:
            pass


class BatchNorm2d(_BatchNorm):
    r"""
    Batch normalization layer over a 4D input.

    Batch Normalization is widely used in convolutional networks. This layer
    applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with
    additional channel dimension) to avoid internal covariate shift as described
    in the paper `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It
    rescales and recenters the feature using a mini-batch of data and
    the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Note:
        The implementation of BatchNorm is different in graph mode and pynative mode, therefore that mode can not be
        changed after net was initialized.
        Note that the formula for updating the running_mean and running_var is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times x_t + \text{momentum} \times \hat{x}`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the new observed value.

    Args:
        num_features (int): `C` from an expected input of size (N, C, H, W).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data. If false,
            use the mean value and variance value of specified value. If None, the training process will use the mean
            and variance of current batch data and track the running mean and variance, the evaluation process will use
            the running mean and variance. Default: None.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.BatchNorm2d(num_features=3)
        >>> np.random.seed(0)
        >>> input = Tensor(np.random.randint(0, 255, [1, 3, 2, 2]), mindspore.float32)
        >>> output = net(input)
        >>> print(output)
        [[[[171.99915   46.999763 ]
           [116.99941  191.99904  ]]
          [[ 66.999664 250.99875  ]
           [194.99902  102.99948  ]]
          [[  8.999955 210.99895  ]
           [ 20.999895 241.9988   ]]]]
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros',
                 moving_mean_init='zeros',
                 moving_var_init='ones',
                 use_batch_statistics=None,
                 data_format='NCHW'):
        super(BatchNorm2d, self).__init__(num_features,
                                          eps,
                                          momentum,
                                          affine,
                                          gamma_init,
                                          beta_init,
                                          moving_mean_init,
                                          moving_var_init,
                                          use_batch_statistics,
                                          input_dims='2d',
                                          data_format=data_format)

    def _check_data_dim(self, x):
        if x.ndim != 4:
            pass


class GlobalBatchNorm(_BatchNorm):
    r"""
    Global normalization layer over a N-dimension input.

    Global Normalization is cross device synchronized batch normalization. The implementation of Batch Normalization
    only normalizes the data within each device. Global normalization will normalize the input within the group.
    It has been described in the paper `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    feature using a mini-batch of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Note:
        Currently, GlobalBatchNorm only supports 2D and 4D inputs.

    Args:
        num_features (int): `C` from an expected input of size (N, C, H, W).
        device_num_each_group (int): The number of devices in each group. Default: 2.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data. If false,
            use the mean value and variance value of specified value. If None, training process will use the mean and
            variance of current batch data and track the running mean and variance, eval process will use the running
            mean and variance. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> # This example should be run with multiple processes.
        >>> # Please refer to the tutorial > Distributed Training on mindspore.cn.
        >>> import numpy as np
        >>> from mindspore.communication import init
        >>> from mindspore import context
        >>> from mindspore.context import ParallelMode
        >>> from mindspore import nn, Tensor
        >>> from mindspore.common import dtype as mstype
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> init()
        >>> context.reset_auto_parallel_context()
        >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)
        >>> np.random.seed(0)
        >>> global_bn_op = nn.GlobalBatchNorm(num_features=3, device_num_each_group=2)
        >>> input = Tensor(np.random.randint(0, 255, [1, 3, 2, 2]), mstype.float32)
        >>> output = global_bn_op(input)
        >>> print(output)
        [[[[171.99915    46.999763]
           [116.99941   191.99904 ]]
          [[ 66.999664  250.99875 ]
           [194.99902   102.99948 ]]
          [[  8.999955  210.99895 ]
           [ 20.9999895 241.9988  ]]]]
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros',
                 moving_mean_init='zeros',
                 moving_var_init='ones',
                 use_batch_statistics=None,
                 device_num_each_group=2):
        super(GlobalBatchNorm, self).__init__(num_features,
                                              eps,
                                              momentum,
                                              affine,
                                              gamma_init,
                                              beta_init,
                                              moving_mean_init,
                                              moving_var_init,
                                              use_batch_statistics,
                                              device_num_each_group,
                                              input_dims='both')
        self.group = validator.check_positive_int(device_num_each_group)
        if self.group <= 1:
            raise ValueError("the number of group must be greater than 1.")

    def _check_data_dim(self, x):
        if x.dim == 0:
            pass


class LayerNorm(Cell):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.

    Layer normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_. Unlike batch
    normalization, layer normalization performs exactly the same computation at training and
    testing time. It can be described using the following formula. It is applied across all channels
    and pixel but only one batch size.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        normalized_shape (Union(tuple[int], list[int]): The normalization is performed over axis
            `begin_norm_axis ... R - 1`.
        begin_norm_axis (int): The first normalization dimension: normalization will be performed along dimensions
            `begin_norm_axis: rank(inputs)`, the value should be in [-1, rank(input)). Default: -1.
        begin_params_axis (int): The first parameter(beta, gamma)dimension: scale and centering parameters
            will have dimensions `begin_params_axis: rank(inputs)` and will be broadcast with
            the normalized inputs accordingly, the value should be in [-1, rank(input)). Default: -1.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.
        epsilon (float): A value added to the denominator for numerical stability. Default: 1e-7.

    Inputs:
        - **input_x** (Tensor) - The shape of 'input_x' is :math:`(x_1, x_2, ..., x_R)`,
          and `input_shape[begin_norm_axis:]` is equal to `normalized_shape`.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
        >>> shape1 = x.shape[1:]
        >>> m = nn.LayerNorm(shape1,  begin_norm_axis=1, begin_params_axis=1)
        >>> output = m(x).shape
        >>> print(output)
        (20, 5, 10, 10)
    """

    def __init__(self,
                 normalized_shape,
                 begin_norm_axis=-1,
                 begin_params_axis=-1,
                 gamma_init='ones',
                 beta_init='zeros',
                 epsilon=1e-7
                 ):
        super(LayerNorm, self).__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError("The type of 'normalized_shape' should be tuple[int] or list[int], but '{}' type is {}."
                            .format(normalized_shape, type(normalized_shape)))
        self.normalized_shape = normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.epsilon = epsilon
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            beta_init, normalized_shape), name="beta")
        self.layer_norm = _selected_ops.LayerNorm(begin_norm_axis=self.begin_norm_axis,
                                                  begin_params_axis=self.begin_params_axis)

    def construct(self, input_x):
        y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        return y

    def extend_repr(self):
        """Display instance object as string."""
        return 'normalized_shape={}, begin_norm_axis={}, begin_params_axis={}, gamma{}, beta={}'.format(
            self.normalized_shape, self.begin_norm_axis, self.begin_params_axis, self.gamma, self.beta)


class GroupNorm(Cell):
    r"""
    Group Normalization over a mini-batch of inputs.

    Group normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_. Group normalization
    divides the channels into groups and computes within each group the mean and variance for normalization,
    and it performs very stable over a wide range of batch size. It can be described using the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        num_groups (int): The number of groups to be divided along the channel dimension.
        num_channels (int): The number of channels per group.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool): A bool value, this layer will have learnable affine parameters when set to true. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'. If gamma_init is a Tensor, the shape must be [num_channels].
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'. If beta_init is a Tensor, the shape must be [num_channels].

    Inputs:
        - **input_x** (Tensor) - The input feature with shape [N, C, H, W].

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> goup_norm_op = nn.GroupNorm(2, 2)
        >>> x = Tensor(np.ones([1, 2, 4, 4], np.float32))
        >>> output = goup_norm_op(x)
        >>> print(output)
        [[[[0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]]
          [[0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]]]]
    """

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, gamma_init='ones', beta_init='zeros'):
        super(GroupNorm, self).__init__()
        self.num_groups = validator.check_positive_int(num_groups)
        self.num_channels = validator.check_positive_int(num_channels)
        if num_channels % num_groups != 0:
            raise ValueError("num_channels should be divided by num_groups")
        self.eps = validator.check_value_type('eps', eps, (float,), type(self).__name__)
        self.affine = validator.check_bool(affine)

        gamma = initializer(gamma_init, num_channels)
        beta = initializer(beta_init, num_channels)
        if self.affine:
            self.gamma = Parameter(gamma, name='gamma')
            self.beta = Parameter(beta, name='beta')
        else:
            self.gamma = gamma
            self.beta = beta
        self.shape = F.shape
        self.reshape = F.reshape
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = F.square
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.sqrt = P.Sqrt()

    def _cal_output(self, x):
        """calculate groupnorm output"""
        batch, channel, height, width = self.shape(x)
        _channel_check(channel, self.num_channels)
        x = self.reshape(x, (batch, self.num_groups, -1))
        mean = self.reduce_mean(x, 2)
        var = self.reduce_sum(self.square(x - mean), 2) / (channel * height * width / self.num_groups)
        std = self.sqrt(var + self.eps)
        x = (x - mean) / std
        x = self.reshape(x, (batch, channel, height, width))
        output = x * self.reshape(self.gamma, (-1, 1, 1)) + self.reshape(self.beta, (-1, 1, 1))
        return output

    def construct(self, x):
        _shape_check(self.shape(x))
        output = self._cal_output(x)
        return output

    def extend_repr(self):
        """Display instance object as string."""
        return 'num_groups={}, num_channels={}'.format(self.num_groups, self.num_channels)
