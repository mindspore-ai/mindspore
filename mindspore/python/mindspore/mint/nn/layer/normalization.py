# Adapted from:
# https://github.com/pytorch/pytorch/blob/release/2.1/torch/nn/modules/batchnorm.py
#
# Additional modifications made by Huawei Technologies Co., Ltd in 2024.
"""normalization for mint"""
from __future__ import absolute_import
from __future__ import division

from typing import Optional
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore import ops
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.nn.layer.normalization import LayerNormExt as LayerNorm
from mindspore.communication import get_group_size
from mindspore.communication._comm_helper import GlobalComm
from mindspore.ops.function import batch_norm

from ._functions import _SyncBatchNorm


class _NormBase(Cell):
    """Common base of _InstanceNorm and _BatchNorm"""

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 dtype=None
                 ) -> None:
        super(_NormBase, self).__init__()
        self.shape = ops.Shape()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.dtype = dtype if dtype is not None else mstype.float32
        if self.affine:
            self.weight = Parameter(
                Tensor(np.empty(num_features), dtype=self.dtype), name="weight")
            self.bias = Parameter(
                Tensor(np.empty(num_features), dtype=self.dtype), name="bias")
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            self.running_mean = Parameter(Tensor(np.zeros(num_features), dtype=self.dtype),
                                          requires_grad=False, name="running_mean")
            self.running_var = Parameter(Tensor(np.ones(num_features), dtype=self.dtype),
                                         requires_grad=False, name="running_var")
            self.num_batches_tracked = Parameter(Tensor(0, dtype=ms.int64),
                                                 requires_grad=False, name="num_batches_tracked")
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        """init parameters"""

        if self.track_running_stats:
            zero_running_mean = Tensor(
                np.zeros(self.num_features), dtype=self.dtype)
            one_running_var = Tensor(
                np.ones(self.num_features), dtype=self.dtype)
            zero_num_batches_tracked = Tensor(0, dtype=ms.int64)

            ops.assign(self.running_mean, zero_running_mean)
            ops.assign(self.running_var, one_running_var)
            ops.assign(self.num_batches_tracked, zero_num_batches_tracked)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            one_weight = Tensor(np.ones(self.num_features), dtype=self.dtype)
            zero_bias = Tensor(np.zeros(self.num_features), dtype=self.dtype)

            ops.assign(self.weight, one_weight)
            ops.assign(self.bias, zero_bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, affine={}, track_running_stats={}'.format(
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)


class _BatchNorm(_NormBase):
    """common base of BatchNormXxx"""

    def __init__(
            self,
            num_features: int,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            dtype=None) -> None:
        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats,
                                         dtype)


    def _check_input_dim(self, input):
        raise NotImplementedError

    def construct(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        return mint.functional.batch_norm(
            input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats
            else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm1d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 2D or 3D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ .

    .. math::

        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    The mean and standard deviation are calculated per-dimension over
    the mini-batches. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default,
    the elements of :math:`\gamma` are set to 1, and the elements of :math:`\beta` are set to 0.

    .. warning::
        This API does not support Dynamic Rank.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, L)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``True`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C)` or :math:`(N, C, L)`,
          where :math:`N` means batch, :math:`C` means the number of feature or the number of channel,
          and :math:`L` is the length of sequence.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]])
        >>> net = mindspore.mint.nn.BatchNorm1d(4)
        >>> output = net(input_x)
        >>> print(output)
        [[0.6999965  0.4999975  0.4999975  0.59999704]
         [0.4999975  0.399998   0.59999704 0.89999545]]
    """

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim != 2 and dim != 3:
            raise ValueError(
                "expected 2D or 3D input, but got " + str(dim) + "D input"
            )


class BatchNorm2d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 4D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard deviation are calculated per-dimension over
    the mini-batches. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default,
    the elements of :math:`\gamma` are set to 1, and the elements of :math:`\beta` are set to 0.

    .. warning::
        - This API does not support Dynamic Rank.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, H, W)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``True`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both train and eval modes. Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, H, W)`.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([0.3, 0.4, 0.5, 0.3])
        >>> input_x = input_x.reshape((2, 2, 1, 1))
        >>> net = mint.nn.BatchNorm2d(2)
        >>> output = net(input_x)
        >>> print(output)
        [[[[0.29999852]]
          [[0.399998  ]]]
         [[[0.4999975 ]]
          [[0.29999852]]]]
    """

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim != 4:
            raise ValueError(
                "expected 4D input, but got " + str(dim) + "D input"
            )


class BatchNorm3d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 5D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard deviation are calculated per-dimension over
    the mini-batches. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default,
    the elements of :math:`\gamma` are set to 1, and the elements of :math:`\beta` are set to 0.

    .. warning::
        This API does not support Dynamic Rank.

    Args:
        num_features (int): `C` from an expected input of shape :math:`(N, C, D, H, W)`.
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average. Default: ``0.1`` .
        affine (bool, optional): a boolean value that when set to ``True``, this cell has
            learnable affine parameters. Default: ``True`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, D, H, W)`.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        ValueError: If `num_features` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([0.1, 0.9, 1.2, 2.3])
        >>> input_x = input_x.reshape((1, 2, 1, 1, 2))
        >>> net = mindspore.mint.nn.BatchNorm3d(2)
        >>> output = net(input_x)
        >>> print(output)
        [[[[[0.0999995  0.89999545]]]
          [[[1.1999941  2.2999885 ]]]]]
    """

    def _check_input_dim(self, input):
        shape = self.shape(input)
        dim = len(shape)
        if dim != 5:
            raise ValueError(
                "expected 5D input, but got " + str(dim) + "D input"
            )


class GroupNorm(Cell):
    r"""
    Group normalization of mini-batch inputs.

    Group Normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_.

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization, and it performs very stable over a wide
    range of batch size. :math:`\gamma` and :math:`\beta` are trainable scale and shift.
    It can be described using the following formula:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is `weight`, :math:`\beta` is `bias`, and :math:`\epsilon` is `eps`.

    Args:
        num_groups (int): The number of groups to be divided along the channel dimension.
        num_channels (int): The number of input channels.
        eps (float, optional): A value added to the denominator for numerical stability. Default: ``1e-05`` .
        affine (bool, optional): The parameters, such as :math:`\gamma` and :math:`\beta`, are learnable
            when set to ``true`` . Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - :math:`(N, C, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        - **output** (Tensor) - the normalized and scaled offset tensor,
          has the same shape and data type as the `input`.

    Raises:
        ValueError: If `num_groups` or `num_channels` is less than 1, or `num_channels`
            is not divided by `num_groups`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> group_norm_op = mindspore.mint.nn.GroupNorm(2, 2)
        >>> x = mindspore.tensor(np.ones([1, 2, 4, 4], np.float32))
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

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, dtype=None):
        """Initialize GroupNorm."""
        super(GroupNorm, self).__init__()
        ms_dtype = mstype.float32 if dtype is None else dtype
        weight_init = 'ones'
        bias_init = 'zeros'

        self.num_groups = validator.check_positive_int(
            num_groups, "num_groups", self.cls_name)
        self.num_channels = validator.check_positive_int(
            num_channels, "num_channels", self.cls_name)
        if num_channels % num_groups != 0:
            raise ValueError(f"For '{self.cls_name}', the 'num_channels' must be divided by 'num_groups', "
                             f"but got 'num_channels': {num_channels}, 'num_groups': {num_groups}.")
        self.eps = validator.check_value_type(
            'eps', eps, (float,), type(self).__name__)
        self.affine = validator.check_bool(
            affine, arg_name="affine", prim_name=self.cls_name)

        self.weight = Parameter(initializer(
            weight_init, self.num_channels, dtype=ms_dtype), name="weight", requires_grad=affine)
        self.bias = Parameter(initializer(
            bias_init, self.num_channels, dtype=ms_dtype), name="bias", requires_grad=affine)

    def _cal_output(self, x):
        """calculate groupnorm output"""
        return ops.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def extend_repr(self):
        return 'num_groups={}, num_channels={}, eps={}, affine={}'.format(
            self.num_groups, self.num_channels, self.eps, self.affine)

    def construct(self, input):
        output = self._cal_output(input)
        return output


class SyncBatchNorm(_BatchNorm):
    r"""
    Sync Batch Normalization layer over a N-dimension input.

    Sync Batch Normalization is cross device synchronized Batch Normalization. In contrast to BN's implementation, 
    which only batches normalizes data in a single device, synchronous BN batches normalizes inputs on all devices 
    within a specified communication group.
    It has been described in the paper `Batch Normalization: Accelerating Deep Network 
    Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    feature using a mini-batch of data and the learned parameters which can be described in the following formula.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. warning::
            This is an experimental API that is subject to change or deletion.

    Args:
        num_features (int): `C` from an expected input of size :math:`(N, C, +)`.
        eps (float, optional): :math:`\epsilon`, a value added to the denominator for numerical stability.
            Default: ``1e-5`` .
        momentum (float, optional): A floating hyperparameter of the momentum for the
            running_mean and running_var computation. Default: ``0.1`` .
        affine (bool, optional): A bool value. When set to ``True`` , :math:`\gamma` and :math:`\beta` are learnable
            parameters. When set to ``False`` , :math:`\gamma` and :math:`\beta` are unlearnable parameters.
            Default: ``True`` .
        track_running_stats (bool, optional): a boolean value that when set to ``True``, this
            cell tracks the running mean and variance, and when set to ``False``,
            this cell does not track such statistics. And this cell always uses batch statistics
            in both training and eval modes. Default: ``True`` .
        process_group (str, optional): synchronization of stats happen within each process group individually.
            Default behavior is synchronization across the whole world. Default: ``None`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, +)`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor, of shape :math:`(N, C, +)`.

    Raises:
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If rank_id in `process_group` is not in range [0, rank_size).

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Here, examples use msrun to pull multi-process distributed tasks across nodes with a single command
            line instruction.
            Please see the `Ascend tutorial
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with multiple devices.

        >>> # Firstly, preparing test_syncbn.py:
        >>> import numpy as np
        >>> import mindspore
        >>> mindspore.communication.init()
        >>> mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE, device_target="Ascend")
        >>> group = "0-1"
        >>> rank_ids = [0, 1]
        >>> mindspore.communication.create_group(group, rank_ids)
        >>> sync_batch_norm = mindspore.mint.nn.layer.SyncBatchNorm(
        ...     num_features=2, process_group=group, dtype=mindspore.float32
        ... )
        >>> sync_batch_norm.set_train(False)
        >>> input_x = mindspore.tensor(np.linspace(0, 5, 2*2*2*2), mindspore.float32).reshape(2, 2, 2, 2)
        >>> output_data = sync_batch_norm(input_x)
        >>> # Then, executing the command such as the following:
        >>> # msrun --worker_num=2 --local_worker_num=2 --master_port=8975 --log_dir=msrun_log --join=True
        >>> # --cluster_time_out=100 pytest -s -v test_syncbn.py

    """
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 process_group: Optional[str] = None,
                 dtype=None):
        super(SyncBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, dtype
        )

        self.process_group = process_group if process_group else GlobalComm.WORLD_COMM_GROUP
        self.world_size = get_group_size(self.process_group)
        self.sync_batch_norm = _SyncBatchNorm(
            self.num_features, self.world_size, self.dtype)

    def _check_input_dim(self, input):
        if input.ndim < 2:
            raise ValueError(
                "expected at least 2D input (got {}D input)".format(input.ndim)
            )

    def _check_non_zero_input_channels(self, input):
        if input.shape[1] == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    def construct(self, input: Tensor) -> Tensor:
        # currently only GPU input is supported

        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked.value())
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        # Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        # Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        # passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        # used for normalization (i.e. in eval mode when buffers are not None).
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (bn_training and self.training)
        if need_sync:
            need_sync = self.world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            if self.weight is None:
                weight = Tensor(np.ones(self.num_features), dtype=self.dtype)
            else:
                weight = self.weight
            if self.bias is None:
                bias = Tensor(np.zeros(self.num_features), dtype=self.dtype)
            else:
                bias = self.bias
            if running_mean is None or running_var is None:
                raise ValueError(
                    "running mean or running var can\'t be none for batch_norm.")
            return batch_norm(input,
                              running_mean,
                              running_var,
                              weight,
                              bias,
                              bn_training,
                              exponential_average_factor,
                              self.eps)
        else:
            output = self.sync_batch_norm(input,
                                          self.weight,
                                          self.bias,
                                          running_mean,
                                          running_var,
                                          self.eps,
                                          exponential_average_factor,
                                          self.process_group,
                                          self.world_size)
            return output


__all__ = [
    'GroupNorm',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
    'SyncBatchNorm',
]
