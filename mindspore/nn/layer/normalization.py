# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import itertools
import numbers

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Initializer
from mindspore.common.tensor import Tensor
from mindspore.common._decorator import deprecated
from mindspore.ops.primitive import constexpr
import mindspore.context as context
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore._extends import cell_attr_register
from mindspore.communication.management import get_group_size, get_rank
from mindspore.communication import management
from mindspore.ops import _selected_ops
from mindspore.common import dtype as mstype
from ..cell import Cell

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm',
           'GlobalBatchNorm', 'SyncBatchNorm', 'InstanceNorm2d']

SYNC_BN_GROUP_NAME = ""

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
                 process_groups=0,
                 input_dims='2d',
                 data_format='NCHW'):
        super(_BatchNorm, self).__init__()
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if momentum < 0 or momentum > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(momentum))
        self.input_dims = input_dims
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
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
        self.group_device_num = validator.check_positive_int(device_num_each_group)
        self.process_groups = process_groups
        self.is_global = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        global SYNC_BN_GROUP_NAME
        # for GlobalBatchNorm
        if self.group_device_num != 1:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            self.device_list = [i for i in range(0, self.rank_size)]
            self.rank_list = self.list_group(self.device_list, self.group_device_num)
            self.rank_list_idx = len(self.rank_list)
            for i in range(self.rank_list_idx):
                if self.rank_id in self.rank_list[i]:
                    self.is_global = True
                    if SYNC_BN_GROUP_NAME == "":
                        SYNC_BN_GROUP_NAME = "sync_bn_group"+ str(i)
                        management.create_group(SYNC_BN_GROUP_NAME, self.rank_list[i])
        # for SyncBatchNorm
        if self.process_groups != 0:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            if self.process_groups is not None:
                validator.check_isinstance("process_groups", self.process_groups, list)
                self._check_rank_ids(self.process_groups, self.rank_size)
                for i in range(len(self.process_groups)):
                    validator.check_isinstance("process_groups[" + str(i) +"]", self.process_groups[i], list)
                    self.group_device_num = len(self.process_groups[i])
                    if self.rank_id in self.process_groups[i] and self.group_device_num > 1:
                        self.is_global = True
                        if SYNC_BN_GROUP_NAME == "":
                            SYNC_BN_GROUP_NAME = "sync_bn_group" + str(i)
                            management.create_group(SYNC_BN_GROUP_NAME, self.process_groups[i])
            elif self.rank_size > 1:
                self.is_global = True
                self.group_device_num = self.rank_size
                self.device_list = [i for i in range(0, self.rank_size)]
                if SYNC_BN_GROUP_NAME == "":
                    SYNC_BN_GROUP_NAME = "sync_bn_group0"
                    management.create_group(SYNC_BN_GROUP_NAME, self.device_list)

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

        self.bn_train = P.BatchNorm(is_training=True,
                                    epsilon=self.eps,
                                    momentum=self.momentum,
                                    data_format=self.format)
        if self.is_global:
            self.bn_train = inner.SyncBatchNorm(epsilon=self.eps,
                                                momentum=self.momentum,
                                                group=SYNC_BN_GROUP_NAME,
                                                device_num=self.group_device_num)

        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format=self.format)

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

    def _check_rank_ids(self, process_groups, rank_size):
        seen = set()
        for rid in itertools.chain(*process_groups):
            validator.check_int_range(rid, 0, rank_size, Rel.INC_LEFT, "rank id in process_groups")
            if rid in seen:
                raise ValueError("rank id in process_groups should not be duplicated.")
            seen.add(rid)

    def construct(self, x):
        _shape_check_bn(self.shape(x), self.input_dims)
        if self.use_batch_statistics is None:
            flag = self.training
        else:
            flag = self.use_batch_statistics

        if flag:
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
    """check input dims of batch norm."""
    dim = len(in_shape)
    if in_dims == '1d' and dim != 2:
        raise ValueError("The input must has 2 dims.")
    if in_dims == '2d' and dim != 4:
        raise ValueError("The input must has 4 dims.")
    if in_dims == '3d' and dim != 5:
        raise ValueError("The input must has 5 dims.")
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
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data. If false,
            use the mean value and variance value of specified value. If None, the training process will use the mean
            and variance of current batch data and track the running mean and variance, the evaluation process will use
            the running mean and variance. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out})`.

    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].

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
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
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

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `data_format` is neither 'NHWC' not 'NCHW'.

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


@constexpr
def _check_3d_shape(input_shape):
    if len(input_shape) != 5:
        raise ValueError("For BatchNorm3d, input data must be 5-dimensional.")


class BatchNorm3d(Cell):
    r"""
    Batch normalization layer over a 5D input.

    Batch Normalization is widely used in convolutional networks. This layer
    applies Batch Normalization over a 5D input (a mini-batch of 3D inputs with
    additional channel dimension) to avoid internal covariate shift.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Note:
        The implementation of BatchNorm is different in graph mode and pynative mode, therefore that mode can not be
        changed after net was initialized.
        Note that the formula for updating the running_mean and running_var is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times x_t + \text{momentum} \times \hat{x}`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the new observed value.

    Args:
        num_features (int): `C` from an expected input of size (N, C, D, H, W).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data. If false,
            use the mean value and variance value of specified value. If None, the training process will use the mean
            and variance of current batch data and track the running mean and variance, the evaluation process will use
            the running mean and variance. Default: None.
        data_format (str): The optional value for data format is 'NCDHW'. Default: 'NCDHW'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, D_{out},H_{out}, W_{out})`.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.BatchNorm3d(num_features=3)
        >>> np.random.seed(0)
        >>> input = Tensor(np.random.randint(0, 255, [16, 3, 10, 32, 32]), mindspore.float32)
        >>> output = net(input)
        >>> print(output.shape)
        (16, 3, 10, 32, 32)
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
                 data_format='NCDHW'):
        super(BatchNorm3d, self).__init__()
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.cls_name)
        self.reshape = P.Reshape()
        self.bn2d = BatchNorm2d(num_features=num_features,
                                eps=eps,
                                momentum=momentum,
                                affine=affine,
                                gamma_init=gamma_init,
                                beta_init=beta_init,
                                moving_mean_init=moving_mean_init,
                                moving_var_init=moving_var_init,
                                use_batch_statistics=use_batch_statistics,
                                data_format="NCHW")

    def construct(self, input_x):
        x_shape = F.shape(input_x)
        _check_3d_shape(x_shape)
        input_x = self.reshape(input_x, (x_shape[0], x_shape[1], x_shape[2]*x_shape[3], x_shape[4]))
        bn2d_out = self.bn2d(input_x)
        bn3d_out = self.reshape(bn2d_out, x_shape)
        return bn3d_out


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

    Raises:
        TypeError: If `num_features` or `device_num_each_group` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `device_num_each_group` is less than 2.

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

    @deprecated("1.2", "SyncBatchNorm", True)
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
        self.group_device_num = validator.check_positive_int(device_num_each_group)
        if self.group_device_num <= 1:
            raise ValueError("the number of group must be greater than 1.")

    def _check_data_dim(self, x):
        if x.dim == 0:
            pass


class SyncBatchNorm(_BatchNorm):
    r"""
    Sync Batch normalization layer over a N-dimension input.

    Sync Batch Normalization is cross device synchronized batch normalization. The implementation of Batch
    Normalization only normalizes the data within each device. Sync Batch normalization will normalize the input
    within the group. It has been described in the paper `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    feature using a mini-batch of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Note:
        Currently, SyncBatchNorm only supports 2D and 4D inputs.

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
            use the mean value and variance value of specified value. If None, training process will use the mean and
            variance of current batch data and track the running mean and variance, eval process will use the running
            mean and variance. Default: None.
        process_groups (list): A list to divide devices into different sync groups, containing N subtraction lists.
            Each subtraction list contains int numbers identifying rank ids which need to be synchronized in the same
            group. All int values must be in [0, rank_size) and different from each other. Default: None, indicating
            synchronization across all devices.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If `process_groups` is not a list.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If rank_id in `process_groups` is not in range [0, rank_size).

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
        >>> sync_bn_op = nn.SyncBatchNorm(num_features=3, process_groups=[[0, 1], [2, 3]])
        >>> input = Tensor(np.random.randint(0, 255, [1, 3, 2, 2]), mstype.float32)
        >>> output = sync_bn_op(input)
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
                 process_groups=None):
        super(SyncBatchNorm, self).__init__(num_features,
                                            eps,
                                            momentum,
                                            affine,
                                            gamma_init,
                                            beta_init,
                                            moving_mean_init,
                                            moving_var_init,
                                            use_batch_statistics,
                                            process_groups=process_groups,
                                            input_dims='both')

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

    Raises:
        TypeError: If `normalized_shape` is neither a list nor tuple.
        TypeError: If `begin_norm_axis` or `begin_params_axis` is not an int.
        TypeError: If `epsilon` is not a float.

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
                                                  begin_params_axis=self.begin_params_axis,
                                                  epsilon=self.epsilon)

    def construct(self, input_x):
        y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        return y

    def extend_repr(self):
        """Display instance object as string."""
        return 'normalized_shape={}, begin_norm_axis={}, begin_params_axis={}, gamma{}, beta={}'.format(
            self.normalized_shape, self.begin_norm_axis, self.begin_params_axis, self.gamma, self.beta)


class InstanceNorm2d(Cell):
    r"""
    Instance normalization layer over a 4D input.

    This layer applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with
    additional channel dimension) as described in the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    \gamma and \beta are learnable parameter vectors of size num_features if affine is True. The standard-deviation
    is calculated via the biased estimator.

    By default, this layer uses instance statistics computed from input data in both training and evaluation modes.

    If use_batch_statistics is set to True, it means training phases, and this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during evaluation. The running estimates are
    kept with a default momentum of 0.1.

    InstanceNorm2d and BatchNorm2d are very similar, but have some differences. InstanceNorm2d is applied on each
    channel of channeled data like RGB images, but BatchNorm2d is usually applied on each batch of batched data.

    Note:
        Note that the formula for updating the running_mean and running_var is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times x_t + \text{momentum} \times \hat{x}`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the new observed value.

    Args:
        num_features (int): `C` from an expected input of size (N, C, H, W).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.1.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data. If false,
            use the mean value and variance value of specified value. Default: True.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C, H, W)`. Data type: float16 or float32.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C, H, W)`. Same type and
        shape as the `input_x`.

    Supported Platforms:
        ``GPU``

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If `momentum` is not a float.
        TypeError: If `affine` is not a bool.
        TypeError: If the type of `gamma_init`/`beta_init`/`moving_mean_init`/`moving_var_init` is not same, or if
            the initialized element type is not float32.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        KeyError: If any of `gamma_init`/`beta_init`/`moving_mean_init`/`moving_var_init` is str and the homonymous
            class inheriting from `Initializer` not exists.

    Examples:
        >>> net = nn.InstanceNorm2d(3)
        >>> np.random.seed(0)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3, 2, 2]), mindspore.float32)
        >>> output = net(input)
        >>> print(output.shape)
        (2, 3, 2, 2)
    """

    @cell_attr_register
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros',
                 moving_mean_init='zeros',
                 moving_var_init='ones',
                 use_batch_statistics=True):
        super(InstanceNorm2d, self).__init__()
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        validator.check_value_type('eps', eps, [float], self.cls_name)
        validator.check_value_type('momentum', momentum, [float], self.cls_name)
        validator.check_value_type('affine', affine, [bool], self.cls_name)
        args_input = {"gamma_init": gamma_init, "beta_init": beta_init,
                      "moving_mean_init": moving_mean_init, "moving_var_init": moving_var_init}
        self.check_types_valid(args_input, 'InstanceNorm2d')
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if momentum < 0 or momentum > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(momentum))
        self.use_batch_statistics = use_batch_statistics
        self.num_features = num_features
        self.eps = eps
        self.input_dims = '2d'
        self.moving_mean = Parameter(initializer(
            moving_mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer(
            moving_var_init, num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)

        self.shape = P.Shape()
        self.momentum = momentum
        self.instance_bn = P.InstanceNorm(is_training=self.use_batch_statistics,
                                          epsilon=self.eps,
                                          momentum=self.momentum)

    def _check_data_dim(self, x):
        raise NotImplementedError

    def construct(self, x):
        _shape_check_bn(self.shape(x), self.input_dims)
        return self.instance_bn(x,
                                self.gamma,
                                self.beta,
                                self.moving_mean,
                                self.moving_variance)[0]

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, gamma={}, beta={}, moving_mean={}, moving_variance={}'.format(
            self.num_features, self.eps, self.momentum, self.gamma, self.beta, self.moving_mean, self.moving_variance)

    def check_types_valid(self, args_dict, name):
        for key, _ in args_dict.items():
            val = args_dict[key]
            if not isinstance(val, (Tensor, numbers.Number, str, Initializer)):
                raise TypeError(f"[{name}]Supported type for arg {key} is [Tensor, numbers.Number, str, Initializer],"
                                f"but got {type(val)}")
            if isinstance(val, Tensor) and val.dtype != mstype.float32:
                raise TypeError(f"[{name}]The type of arg {key} should be float32, but got {val.dtype}")


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

    Raises:
        TypeError: If `num_groups` or `num_channels` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If `affine` is not a bool.
        ValueError: If `num_groups` or `num_channels` is less than 1.
        ValueError: If `num_channels` is not divided by `num_groups`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
