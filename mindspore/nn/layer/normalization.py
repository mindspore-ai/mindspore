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
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore._checkparam import check_bool, check_typename
from mindspore._extends import cell_attr_register
from mindspore.communication.management import get_group_size, get_rank
from mindspore.communication import management
from mindspore._checkparam import check_int_positive
from ..cell import Cell


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
                 use_batch_statistics=True,
                 group=1):
        super(_BatchNorm, self).__init__()
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if momentum < 0 or momentum > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(momentum))

        self.use_batch_statistics = use_batch_statistics
        self.num_features = num_features
        self.eps = eps
        self.moving_mean = Parameter(initializer(
            moving_mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer(
            moving_var_init, num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)
        self.group = check_int_positive(group)
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
        self.reduce_mean = P.ReduceMean()
        self.square = P.Square()

        if context.get_context("enable_ge"):
            self.is_ge_backend = True
            self.momentum = Tensor(1.0 - momentum, mstype.float32)
            self.bn_train = P.BatchNorm(is_training=True,
                                        epsilon=self.eps)
        else:
            self.is_ge_backend = False
            self.momentum = 1.0 - momentum
            self.bn_train = P.FusedBatchNorm(mode=1,
                                             epsilon=self.eps,
                                             momentum=self.momentum)
        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps)

        data_parallel_strategy = ((1,), (1,))
        data_parallel_strategy_one = ((1,), ())
        self.sub_mean = P.Sub().set_strategy(data_parallel_strategy)
        self.sub_var = P.Sub().set_strategy(data_parallel_strategy)
        self.mul_mean = P.Mul().set_strategy(data_parallel_strategy_one)
        self.mul_var = P.Mul().set_strategy(data_parallel_strategy_one)
        self.assign_sub_mean = P.AssignSub().set_strategy(data_parallel_strategy)
        self.assign_sub_var = P.AssignSub().set_strategy(data_parallel_strategy)

    def _check_data_dim(self, x):
        raise NotImplementedError

    def list_group(self, world_rank, group_size):
        if group_size > get_group_size():
            raise ValueError("group size can not be greater than local rank size, group size is {}, "
                             "local_rank_size is {}".format(group_size, get_group_size()))
        if len(world_rank) % group_size != 0:
            raise ValueError("please make your group size correct.")
        world_rank_list = zip(*(iter(world_rank),) *group_size)
        group_list = [list(i) for i in world_rank_list]
        return group_list

    def construct(self, x):
        if self.training and self.use_batch_statistics:
            if self.is_ge_backend:
                if self.is_global:
                    x_mean = self.reduce_mean(x)
                    x_mean_square = self.reduce_mean(self.square(x))
                    global_batch_mean = self.all_reduce(x_mean) / self.group
                    global_batch_mean_square = self.all_reduce(x_mean_square) / self.group
                    global_mean = global_batch_mean
                    global_var = global_batch_mean_square - self.square(global_batch_mean)
                    y, batch_mean, batch_var, _, _ = \
                        self.bn_train(x,
                                      self.gamma,
                                      self.beta,
                                      None,
                                      None)

                    mean_sub = self.sub_mean(self.moving_mean, global_mean)
                    temp_mean = self.mul_mean(mean_sub, self.momentum)
                    mean_sub2 = self.sub_var(self.moving_variance, global_var)
                    temp_variance = self.mul_var(mean_sub2, self.momentum)
                    y = F.depend(y, self.assign_sub_mean(self.moving_mean, temp_mean))
                    y = F.depend(y, self.assign_sub_var(self.moving_variance, temp_variance))
                else:
                    y, batch_mean, batch_var, _, _ = \
                        self.bn_train(x,
                                      self.gamma,
                                      self.beta,
                                      None,
                                      None)

                    mean_sub = self.sub_mean(self.moving_mean, batch_mean)
                    temp_mean = self.mul_mean(mean_sub, self.momentum)
                    mean_sub2 = self.sub_var(self.moving_variance, batch_var)
                    temp_variance = self.mul_var(mean_sub2, self.momentum)
                    y = F.depend(y, self.assign_sub_mean(self.moving_mean, temp_mean))
                    y = F.depend(y, self.assign_sub_var(self.moving_variance, temp_variance))
            else:
                y = self.bn_train(x,
                                  self.gamma,
                                  self.beta,
                                  self.moving_mean,
                                  self.moving_variance)[0]
        else:
            y = self.bn_infer(x,
                              self.gamma,
                              self.beta,
                              self.moving_mean,
                              self.moving_variance)[0]
        return y

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, gamma={}, beta={}, moving_mean={}, moving_variance={}'.format(
            self.num_features, self.eps, self.momentum, self.gamma, self.beta, self.moving_mean, self.moving_variance)


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

    Args:
        num_features (int): `C` from an expected input of size (N, C).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.9.
        affine (bool): A bool value when set to True, gamma and beta can be learnable. Default: True.
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
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data, else use
            the mean value and variance value of specified value. Default: True.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = nn.BatchNorm1d(num_features=16)
        >>> input = Tensor(np.random.randint(0, 255, [3, 16]), mindspore.float32)
        >>> net(input)
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
                 use_batch_statistics=True):
        super(BatchNorm1d, self).__init__(num_features,
                                          eps,
                                          momentum,
                                          affine,
                                          gamma_init,
                                          beta_init,
                                          moving_mean_init,
                                          moving_var_init,
                                          use_batch_statistics)
    def _check_data_dim(self, x):
        if x.dim() != 2:
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

    Args:
        num_features (int): `C` from an expected input of size (N, C, H, W).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.9.
        affine (bool): A bool value when set to True, gamma and beta can be learnable. Default: True.
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
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data, else use
            the mean value and variance value of specified value. Default: True.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = nn.BatchNorm2d(num_features=3)
        >>> input = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), mindspore.float32)
        >>> net(input)
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
                 use_batch_statistics=True):
        super(BatchNorm2d, self).__init__(num_features,
                                          eps,
                                          momentum,
                                          affine,
                                          gamma_init,
                                          beta_init,
                                          moving_mean_init,
                                          moving_var_init,
                                          use_batch_statistics)
    def _check_data_dim(self, x):
        if x.dim() != 4:
            pass


class GlobalBatchNorm(_BatchNorm):
    r"""
    Global normalization layer over a N-dimension input.

    Global Normalization is cross device synchronized batch normalization. Batch Normalization implementation
    only normalize the data within each device. Global normalization will normalize the input within the group.
    It has been described in the paper `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    feature using a mini-batch of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        num_features (int): `C` from an expected input of size (N, C, H, W).
        group (int): The number of device in each group.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.9.
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
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data, else use
            the mean value and variance value of specified value. Default: True.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> global_bn_op = nn.GlobalBatchNorm(num_features=3, group=4)
        >>> input = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), mindspore.float32)
        >>> global_bn_op(input)
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
                 use_batch_statistics=True,
                 group=1):
        super(GlobalBatchNorm, self).__init__(num_features,
                                                eps,
                                                momentum,
                                                affine,
                                                gamma_init,
                                                beta_init,
                                                moving_mean_init,
                                                moving_var_init,
                                                use_batch_statistics,
                                                group)
        self.group = check_int_positive(group)
        if self.group <= 1:
            raise ValueError("the number of group must be greater than 1.")
    def _check_data_dim(self, x):
        if x.dim == 0:
            pass

class LayerNorm(Cell):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.

    Layer normalization is widely used in recurrent neural networks. It applies
    normalization over a mini-batch of inputs for each single training case as described
    in the paper `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_. Unlike batch
    normalization, layer normalization performs exactly the same computation at training and
    testing times. It can be described using the following formula. It is applied across all channels
    and pixel but only one batch size.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        normalized_shape (Union(tuple[int], list[int]): The normalization is performed over axes
            `begin_norm_axis ... R - 1` and centering and scaling parameters are calculated over
            `begin_params_axis ... R - 1`.
        begin_norm_axis (int): It first normalization dimension: normalization will be performed along dimensions
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

    Inputs:
        - **input_x** (Tensor) - The shape of 'input_x' is :math:`(x_1, x_2, ..., x_R)`,
          and `input_shape[begin_norm_axis:]` is equal to `normalized_shape`.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `input_x`.

    Examples:
        >>> x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
        >>> shape1 = x.shape()[1:]
        >>> m = nn.LayerNorm(shape1,  begin_norm_axis=1, begin_params_axis=1)
        >>> m(x)
    """
    def __init__(self,
                 normalized_shape,
                 begin_norm_axis=-1,
                 begin_params_axis=-1,
                 gamma_init='ones',
                 beta_init='zeros',
                 ):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            beta_init, normalized_shape), name="beta")
        self.layer_norm = P.LayerNorm(begin_norm_axis=self.begin_norm_axis, begin_params_axis=self.begin_params_axis)

    def construct(self, input_x):
        y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        return y

    def extend_repr(self):
        """Display instance object as string."""
        s = 'normalized_shape={}, begin_norm_axis={}, begin_params_axis={}, gamma{}, beta={}'.format(
            self.normalized_shape, self.begin_norm_axis, self.begin_params_axis, self.gamma, self.beta)
        return s

class GroupNorm(Cell):
    r"""
    Group Normalization over a mini-batch of inputs.

    Group normalization is widely used in recurrent neural networks. It applies
    normalization over a mini-batch of inputs for each single training case as described
    in the paper `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_. Group normalization
    divides the channels into groups and computes within each group the mean and variance for normalization,
    and it performs very stable over a wide range of batch size. It can be described using the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        num_groups (int): The number of groups to be divided along the channel dimension.
        num_channels (int): The number of channels per group.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool): A bool value, this layer will has learnable affine parameters when set to true. Default: True.

    Inputs:
        - **input_x** (Tensor) - The input feature with shape [N, C, H, W].

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `input_x`.

    Examples:
        >>> goup_norm_op = nn.GroupNorm(16, 64)
        >>> x = Tensor(np.ones([1, 64, 256, 256], np.float32))
        >>> goup_norm_op(x)
    """
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = check_int_positive(num_groups)
        self.num_channels = check_int_positive(num_channels)
        if num_channels % num_groups != 0:
            raise ValueError("num_channels should be divided by num_groups")
        self.eps = Tensor(check_typename('eps', eps, (float,)), mstype.float32)
        self.affine = check_bool(affine)

        gamma = initializer('ones', [num_channels, 1, 1], mstype.float32)
        beta = initializer('zeros', [num_channels, 1, 1], mstype.float32)
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

    def construct(self, x):
        batch, channel, height, width = self.shape(x)
        x = self.reshape(x, (batch, self.num_groups, channel*height*width/self.num_groups))
        mean = self.reduce_mean(x, 2)
        var = self.reduce_sum(self.square(x - mean), 2) / (channel * height * width / self.num_groups - 1)
        std = self.sqrt(var + self.eps)
        x = (x - mean) / std
        x = self.reshape(x, (batch, channel, height, width))
        output = x * self.gamma + self.beta
        return output

    def extend_repr(self):
        """Display instance object as string."""
        s = 'num_groups={}, num_channels={}'.format(self.num_groups, self.num_channels)
        return s
