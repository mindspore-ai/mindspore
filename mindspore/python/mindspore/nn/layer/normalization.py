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
from __future__ import absolute_import
from __future__ import division

import itertools
import numbers
import hashlib

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Initializer
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr
import mindspore.context as context
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore._extends import cell_attr_register
from mindspore.communication.management import get_group_size, get_rank
from mindspore.communication import management
from mindspore.common import dtype as mstype
from mindspore.parallel._utils import _is_in_auto_parallel_mode
from mindspore.nn.cell import Cell
from mindspore import log as logger

__all__ = ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'GroupNorm',
           'SyncBatchNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d']


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
                 data_format='NCHW'):
        """Initialize _BatchNorm."""
        super(_BatchNorm, self).__init__()
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        if num_features < 1:
            raise ValueError(f"For '{self.cls_name}', the 'num_features' must be at least 1, but got {num_features}.")

        if momentum < 0 or momentum > 1:
            raise ValueError(f"For '{self.cls_name}', the 'momentum' must be a number in range [0, 1], "
                             f"but got {momentum}.")
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.cls_name}', the 'NHWC' format only support in GPU target, but got device "
                             f"target {context.get_context('device_target')}.")
        self.use_batch_statistics = use_batch_statistics
        if self.use_batch_statistics is not None and not isinstance(self.use_batch_statistics, bool):
            raise ValueError(f"For '{self.cls_name}', the 'use_batch_statistics' must be a boolean value or None,"
                             f" but got {use_batch_statistics}.")
        self.num_features = num_features
        self.eps = eps
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.moving_mean = Parameter(initializer(
            moving_mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer(
            moving_var_init, num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)

        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")

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

        self.bn_train = P.BatchNorm(is_training=True,
                                    epsilon=self.eps,
                                    momentum=self.momentum,
                                    data_format=self.format)

        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format=self.format)
        if _is_in_auto_parallel_mode():
            data_parallel_strategy = ((1,), (1,))
            data_parallel_strategy_one = ((1,), ())
        else:
            data_parallel_strategy = None
            data_parallel_strategy_one = None
        self.sub_mean = P.Sub().shard(data_parallel_strategy)
        self.sub_var = P.Sub().shard(data_parallel_strategy)
        self.mul_mean = P.Mul().shard(data_parallel_strategy_one)
        self.mul_var = P.Mul().shard(data_parallel_strategy_one)
        self.assign_sub_mean = P.AssignSub().shard(data_parallel_strategy)
        self.assign_sub_var = P.AssignSub().shard(data_parallel_strategy)


    def construct(self, x):
        if self.use_batch_statistics is None:
            if self.training:
                return self.bn_train(x,
                                     self.gamma,
                                     self.beta,
                                     self.moving_mean,
                                     self.moving_variance)[0]
            if not self.training:
                return self.bn_infer(x,
                                     self.gamma,
                                     self.beta,
                                     self.moving_mean,
                                     self.moving_variance)[0]

        if self.use_batch_statistics:
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
            self.num_features, self.eps, 1.0 - self.momentum, self.gamma, self.beta, \
            self.moving_mean, self.moving_variance)


class BatchNorm1d(_BatchNorm):
    r"""
    This layer
    applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D or 2D inputs) to
    reduce internal covariate shift. Batch Normalization is widely used in convolutional networks.
    For the setailed contents, refer to `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It
    rescales and recenters the feature using a mini-batch of data and
    the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Note:
        The implementation of BatchNorm is different in graph mode and pynative mode, therefore the mode is not
        recommended to be changed after net was initialized.

    Args:
        num_features (int): number of features or channels `C` of the input `x` .
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
        - **x** (Tensor) - Tensor of shape :math:`(N, C)` or :math:`(N, C, L)` ,
          where `N` is the batch size, `C` is the number of features or channels, and `L` is the sequence length.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C)` or :math:`(N, C, L)` .

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.BatchNorm1d(num_features=4)
        >>> x = Tensor(np.array([[0.7, 0.5, 0.5, 0.6],
        ...                      [0.5, 0.4, 0.6, 0.9]]).astype(np.float32))
        >>> output = net(x)
        >>> print(output)
        [[ 0.6999965   0.4999975  0.4999975  0.59999704 ]
         [ 0.4999975   0.399998   0.59999704 0.89999545 ]]
    """


class BatchNorm2d(_BatchNorm):
    r"""
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
        Note that the formula for updating the :math:`moving\_mean` and :math:`moving\_var` is

        .. math::
            \text{moving_mean}=\text{moving_mean∗momentum}+μ_β\text{∗(1−momentum)}\\
            \text{moving_var}=\text{moving_var∗momentum}+σ^2_β\text{∗(1−momentum)}

        where :math:`moving\_mean` is the updated mean, :math:`moving\_var` is the updated variance,
        :math:`μ_β, σ^2_β` are the observed value (mean and variance) of each batch of data.

    Args:
        num_features (int): The number of channels of the input tensor. Expected input size is (N, C, H, W),
            `C` represents the number of channels.
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
        use_batch_statistics (bool):

            - If true, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If false, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to true or false according to the training
              and evaluation mode. During training, the parameter is set to true, and during evaluation, the
              parameter is set to false. Default: None.

        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, H, W)`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C, H, W)`.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `data_format` is neither 'NHWC' not 'NCHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.BatchNorm2d(num_features=3)
        >>> x = Tensor(np.ones([1, 3, 2, 2]).astype(np.float32))
        >>> output = net(x)
        >>> print(output)
        [[[[ 0.999995 0.999995 ]
           [ 0.999995 0.999995 ]]
          [[ 0.999995 0.999995 ]
           [ 0.999995 0.999995 ]]
          [[ 0.999995 0.999995 ]
           [ 0.999995 0.999995 ]]]]
    """


class BatchNorm3d(Cell):
    r"""
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

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C_{out}, D_{out},H_{out}, W_{out})`.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.BatchNorm3d(num_features=3)
        >>> x = Tensor(np.ones([16, 3, 10, 32, 32]).astype(np.float32))
        >>> output = net(x)
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
                 use_batch_statistics=None):
        """Initialize BatchNorm3d."""
        super(BatchNorm3d, self).__init__()
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
        self.shape = P.Shape()
        self.reshape = P.Reshape()


    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))
        bn2d_out = self.bn2d(x)
        bn3d_out = self.reshape(bn2d_out, x_shape)
        return bn3d_out


SYNCBN_GROUP_DICT = None


def _syncbatchnorm_group_dict():
    global SYNCBN_GROUP_DICT
    if SYNCBN_GROUP_DICT is None:
        SYNCBN_GROUP_DICT = dict()
    return SYNCBN_GROUP_DICT


class SyncBatchNorm(_BatchNorm):
    r"""
    Sync Batch Normalization layer over a N-dimension input.

    Sync Batch Normalization is cross device synchronized Batch Normalization. The implementation of Batch
    Normalization only normalizes the data within each device. Sync Batch Normalization will normalize the input
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
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

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
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#preparations>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `GPU tutorial
            <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#preparation>`_ .

            This example should be run with multiple devices.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> from mindspore import nn
        >>> from mindspore import dtype as mstype
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> ms.reset_auto_parallel_context()
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        >>> sync_bn_op = nn.SyncBatchNorm(num_features=3, process_groups=[[0, 1], [2, 3]])
        >>> x = Tensor(np.ones([1, 3, 2, 2]), mstype.float32)
        >>> output = sync_bn_op(x)
        >>> print(output)
        [[[[ 0.999995 0.999995 ]
           [ 0.999995 0.999995 ]]
          [[ 0.999995 0.999995 ]
           [ 0.999995 0.999995 ]]
          [[ 0.999995 0.999995 ]
           [ 0.999995 0.999995 ]]]]
    """
    @cell_attr_register(attrs=['num_features', 'process_groups'])
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
        """Initialize SyncBatchNorm."""
        super(SyncBatchNorm, self).__init__(num_features,
                                            eps,
                                            momentum,
                                            affine,
                                            gamma_init,
                                            beta_init,
                                            moving_mean_init,
                                            moving_var_init,
                                            use_batch_statistics)
        self.is_global = False
        self.group_name = None
        self.process_groups = process_groups
        if self.process_groups != 0:
            self.is_global = True
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            if self.process_groups is not None:
                validator.check_isinstance("process_groups", self.process_groups, list)
                self._check_rank_ids(self.process_groups, self.rank_size)
                self._create_sync_groups()
            elif self.rank_size > 1:
                self.group_device_num = self.rank_size
                if context.get_context("device_target") == "Ascend":
                    self.group_name = "hccl_world_group"
                elif context.get_context("device_target") == "GPU":
                    self.group_name = "nccl_world_group"

        if self.is_global:
            self.bn_train = inner.SyncBatchNorm(epsilon=self.eps,
                                                momentum=self.momentum,
                                                group=self.group_name,
                                                device_num=self.group_device_num)

    def _create_sync_groups(self):
        """ create groups by process groups. """
        for sub_group in self.process_groups:
            validator.check_isinstance("sub group", sub_group, list)
            self.group_device_num = len(sub_group)
            if self.rank_id in sub_group and self.group_device_num > 1:
                rank_list_name = '_'.join('%s' % id for id in sub_group)
                group_dict = _syncbatchnorm_group_dict()
                if rank_list_name not in group_dict:
                    md5 = hashlib.md5()
                    md5.update(rank_list_name.encode('utf-8'))
                    hash_name = md5.hexdigest()
                    self.group_name = str(self.group_device_num) + '_' + hash_name
                    group_dict[rank_list_name] = self.group_name
                    management.create_group(self.group_name, sub_group)
                    logger.info("create group for sync batchnorm, the rank list is {}, the group name is {}".format(
                        rank_list_name, self.group_name))
                else:
                    self.group_name = group_dict[rank_list_name]
                    logger.info("the group for {} already exists, no need to create".format(rank_list_name))

    def _check_rank_ids(self, process_groups, rank_size):
        seen = set()
        for rid in itertools.chain(*process_groups):
            validator.check_int_range(rid, 0, rank_size, Rel.INC_LEFT, "rank id in process_groups", self.cls_name)
            if rid in seen:
                raise ValueError(f"For '{self.cls_name}', rank id in 'process_groups' must not be duplicated, "
                                 f"but got {process_groups}.")
            seen.add(rid)


class LayerNorm(Cell):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.

    Layer Normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_. Unlike Batch
    Normalization, Layer Normalization performs exactly the same computation at training and
    testing time. It is applied across all channels
    and pixel but only one batch size. It can be described using the following formula:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        normalized_shape (Union(tuple[int], list[int])): The normalization is performed over axis
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
        - **x** (Tensor) - The shape of `x` is :math:`(x_1, x_2, ..., x_R)`,
          and `input_shape[begin_norm_axis:]` is equal to `normalized_shape`.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `x`.

    Raises:
        TypeError: If `normalized_shape` is neither a list nor tuple.
        TypeError: If `begin_norm_axis` or `begin_params_axis` is not an int.
        TypeError: If `epsilon` is not a float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        """Initialize LayerNorm."""
        super(LayerNorm, self).__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError(f"For '{self.cls_name}', the type of 'normalized_shape' must be tuple[int] or list[int], "
                            f"but got {normalized_shape} and the type is {type(normalized_shape)}.")
        self.normalized_shape = normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.epsilon = epsilon
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            beta_init, normalized_shape), name="beta")
        self.layer_norm = P.LayerNorm(begin_norm_axis=self.begin_norm_axis,
                                      begin_params_axis=self.begin_params_axis,
                                      epsilon=self.epsilon)

    def construct(self, input_x):
        y, _, _ = self.layer_norm(input_x, self.gamma.astype(input_x.dtype), self.beta.astype(input_x.dtype))
        return y

    def extend_repr(self):
        return 'normalized_shape={}, begin_norm_axis={}, begin_params_axis={}, gamma{}, beta={}'.format(
            self.normalized_shape, self.begin_norm_axis, self.begin_params_axis, self.gamma, self.beta)


class _InstanceNorm(Cell):
    """Instance Normalization base class."""
    @cell_attr_register
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros'):
        """Initialize Normalization base class."""
        super(_InstanceNorm, self).__init__()
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        validator.check_value_type('eps', eps, [float], self.cls_name)
        validator.check_value_type('momentum', momentum, [float], self.cls_name)
        validator.check_value_type('affine', affine, [bool], self.cls_name)
        args_input = {"gamma_init": gamma_init, "beta_init": beta_init}
        self.check_types_valid(args_input, 'InstanceNorm2d')
        if num_features < 1:
            raise ValueError(f"For '{self.cls_name}', the 'num_features' must be at least 1, but got {num_features}.")

        if momentum < 0 or momentum > 1:
            raise ValueError(f"For '{self.cls_name}', the 'momentum' must be a number in range [0, 1], "
                             f"but got {momentum}.")
        self.num_features = num_features
        self.eps = eps
        self.moving_mean = Parameter(initializer('zeros', num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer('ones', num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)

        self.shape = P.Shape()
        self.momentum = momentum
        self.instance_bn = P.InstanceNorm(epsilon=self.eps, momentum=self.momentum)

    def construct(self, x):
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
                raise TypeError(f"For '{self.cls_name}', the type of '{key}' must be in "
                                f"[Tensor, numbers.Number, str, Initializer], but got type {type(val).__name__}.")
            if isinstance(val, Tensor) and val.dtype != mstype.float32:
                raise TypeError(f"For '{self.cls_name}', the type of '{key}' must be float32, "
                                f"but got {val.dtype}.")


class InstanceNorm1d(_InstanceNorm):
    r"""
    This layer applies Instance Normalization over a 3D input (a mini-batch of 1D inputs with
    additional channel dimension). Refer to the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The size of :math:`\gamma` and :math:`\beta`, learnable parameters vectors, is num_features if affine is True.
    The standard-deviation is calculated via the biased estimator.

    This layer uses instance statistics computed from input data in both training and evaluation modes.

    InstanceNorm1d and BatchNorm1d are very similar, but have some differences. InstanceNorm1d is applied on each
    channel of channeled data like RGB images, but BatchNorm1d is usually applied on each batch of batched data.

    Note:
        Note that the formula for updating the running_mean and running_var is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times x_t + \text{momentum} \times \hat{x}`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the new observed value.

    Args:
        num_features (int): `C` from an expected input of size (N, C, L).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.1.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
            When initialized with Tensor, the shape should be :math:`(C)`. Default: 'zeros'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
            When initialized with Tensor, the shape should be :math:`(C)`. Default: 'zeros'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, L)`. Data type: float16 or float32.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C, L)`. Same type and
        shape as the `x`.

    Raises:
        TypeError: If the type of `num_features` is not int.
        TypeError: If the type of `eps` is not float.
        TypeError: If the type of `momentum` is not float.
        TypeError: If the type of `affine` is not bool.
        TypeError: If the type of `gamma_init`/`beta_init` is not same, or if the initialized element type is not
            float32.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If the shape of `gamma_init` / `beta_init` is not :math:`(C)`.
        KeyError: If any of `gamma_init`/`beta_init` is str and the homonymous class inheriting from `Initializer` not
            exists.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.InstanceNorm1d(3)
        >>> x = Tensor(np.ones([2, 3, 5]), mindspore.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (2, 3, 5)
    """


class InstanceNorm2d(_InstanceNorm):
    r"""
    This layer applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with
    additional channel dimension). Refer to the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable parameter vectors of size num_features if affine is True.
    The standard-deviation is calculated via the biased estimator.

    This layer uses instance statistics computed from input data in both training and evaluation modes.

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
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
            When initialized with Tensor, the shape should be :math:`(C)`. Default: 'zeros'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
            When initialized with Tensor, the shape should be :math:`(C)`. Default: 'zeros'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, H, W)`. Data type: float16 or float32.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C, H, W)`. Same type and
        shape as the `x`.

    Raises:
        TypeError: If the type of `num_features` is not int.
        TypeError: If the type of `eps` is not float.
        TypeError: If the type of `momentum` is not float.
        TypeError: If the type of `affine` is not bool.
        TypeError: If the type of `gamma_init`/`beta_init` is not same, or if the initialized element type is not
            float32.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If the shape of `gamma_init` / `beta_init` is not :math:`(C)`.
        KeyError: If any of `gamma_init`/`beta_init` is str and the homonymous class inheriting from `Initializer` not
            exists.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.InstanceNorm2d(3)
        >>> x = Tensor(np.ones([2, 3, 2, 2]), mindspore.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (2, 3, 2, 2)
    """


class InstanceNorm3d(_InstanceNorm):
    r"""
    This layer applies Instance Normalization over a 5D input (a mini-batch of 3D inputs with
    additional channel dimension). Refer to the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable parameter vectors of size num_features if affine is True.
    The standard-deviation is calculated via the biased estimator.

    This layer uses instance statistics computed from input data in both training and evaluation modes.

    InstanceNorm3d and BatchNorm3d are very similar, but have some differences. InstanceNorm3d is applied on each
    channel of channeled data like RGB images, but BatchNorm3d is usually applied on each batch of batched data.

    Note:
        Note that the formula for updating the running_mean and running_var is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times x_t + \text{momentum} \times \hat{x}`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the new observed value.

    Args:
        num_features (int): `C` from an expected input of size (N, C, D, H, W).
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: 0.1.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
            When initialized with Tensor, the shape should be :math:`(C)`. Default: 'zeros'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
            When initialized with Tensor, the shape should be :math:`(C)`. Default: 'zeros'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, D, H, W)`. Data type: float16 or float32.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C, D, H, W)`. Same type and
        shape as the `x`.

    Raises:
        TypeError: If the type of `num_features` is not int.
        TypeError: If the type of `eps` is not float.
        TypeError: If the type of `momentum` is not float.
        TypeError: If the type of `affine` is not bool.
        TypeError: If the type of `gamma_init`/`beta_init` is not same, or if the initialized element type is not
            float32.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If the shape of `gamma_init` / `beta_init` is not :math:`(C)`.
        KeyError: If any of `gamma_init`/`beta_init` is str and the homonymous class inheriting from `Initializer` not
            exists.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> net = nn.InstanceNorm3d(3)
        >>> x = Tensor(np.ones([2, 3, 5, 2, 2]), mindspore.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (2, 3, 5, 2, 2)
    """


class GroupNorm(Cell):
    r"""
    Group Normalization over a mini-batch of inputs.

    Group Normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_. Group Normalization
    divides the channels into groups and computes within each group the mean and variance for normalization,
    and it performs very stable over a wide range of batch size. It can be described using the following formula:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        num_groups (int): The number of groups to be divided along the channel dimension.
        num_channels (int): The number of input channels.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool): A bool value, this layer will have learnable affine parameters when set to true. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'. If gamma_init is a Tensor, the shape must be [num_channels].
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'. If beta_init is a Tensor, the shape must be [num_channels].

    Inputs:
        - **x** (Tensor) - The input feature with shape :math:`(N, C, H, W)` .

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `x`.

    Raises:
        TypeError: If `num_groups` or `num_channels` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If `affine` is not a bool.
        ValueError: If `num_groups` or `num_channels` is less than 1.
        ValueError: If `num_channels` is not divided by `num_groups`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> group_norm_op = nn.GroupNorm(2, 2)
        >>> x = Tensor(np.ones([1, 2, 4, 4], np.float32))
        >>> output = group_norm_op(x)
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
        """Initialize GroupNorm."""
        super(GroupNorm, self).__init__()
        self.num_groups = validator.check_positive_int(num_groups, "num_groups", self.cls_name)
        self.num_channels = validator.check_positive_int(num_channels, "num_channels", self.cls_name)
        if num_channels % num_groups != 0:
            raise ValueError(f"For '{self.cls_name}', the 'num_channels' must be divided by 'num_groups', "
                             f"but got 'num_channels': {num_channels}, 'num_groups': {num_groups}.")
        self.eps = validator.check_value_type('eps', eps, (float,), type(self).__name__)
        self.affine = validator.check_bool(affine, arg_name="affine", prim_name=self.cls_name)

        self.gamma = Parameter(initializer(
            gamma_init, num_channels), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_channels), name="beta", requires_grad=affine)
        self.shape = F.shape
        self.reshape = F.reshape
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = F.square
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.sqrt = P.Sqrt()

    def _cal_output(self, x):
        """calculate groupnorm output"""
        batch, channel, height, width = self.shape(x)
        x = self.reshape(x, (batch, self.num_groups, -1))
        mean = self.reduce_mean(x, 2)
        var = self.reduce_sum(self.square(x - mean), 2) / (channel * height * width / self.num_groups)
        std = self.sqrt(var + self.eps)
        x = (x - mean) / std
        x = self.reshape(x, (batch, channel, height, width))
        output = x * self.reshape(self.gamma, (-1, 1, 1)) + self.reshape(self.beta, (-1, 1, 1))
        return output

    @staticmethod
    @constexpr
    def _check_dtype(dtype, valid_dtypes, prim_name=None):
        validator.check_type_name("input", dtype, valid_dtypes, prim_name)

    def extend_repr(self):
        return 'num_groups={}, num_channels={}'.format(self.num_groups, self.num_channels)

    def construct(self, x):
        self._check_dtype(x.dtype, [mstype.float16, mstype.float32], self.cls_name)
        output = self._cal_output(x)
        return output
