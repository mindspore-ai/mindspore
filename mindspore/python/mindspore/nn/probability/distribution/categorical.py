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
"""Categorical Distribution"""
import numpy as np
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops.functional import stop_gradient
from mindspore.ops.operations import _inner_ops as inner
from mindspore._checkparam import Validator
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_prob, check_sum_equal_one, check_rank,\
    check_distribution_name
from ._utils.custom_ops import exp_generic, log_generic, broadcast_to


class Categorical(Distribution):
    r"""
    Categorical distribution.
    A Categorical Distribution is a discrete distribution with the range :math:`\{1, 2, ..., k\}`
    and the probability mass function as :math:`P(X = i) = p_i, i = 1, ..., k`.

    Args:
        probs (Tensor, list, numpy.ndarray): Event probabilities. Default: None.
        seed (int): The global seed is used in sampling. Global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.int32.
        name (str): The name of the distribution. Default: Categorical.

    Note:
        `probs` must have rank at least 1, values are proper probabilities and sum to 1.

    Raises:
        ValueError: When the sum of all elements in `probs` is not 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> # To initialize a Categorical distribution of probs [0.5, 0.5]
        >>> ca1 = msd.Categorical(probs=[0.2, 0.8], dtype=mindspore.int32)
        >>> # A Categorical distribution can be initialized without arguments.
        >>> # In this case, `probs` must be passed in through arguments during function calls.
        >>> ca2 = msd.Categorical(dtype=mindspore.int32)
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1, 0], dtype=mindspore.int32)
        >>> probs_a = Tensor([0.5, 0.5], dtype=mindspore.float32)
        >>> probs_b = Tensor([0.35, 0.65], dtype=mindspore.float32)
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, are the same as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     probs (Tensor): event probabilities. Default: self.probs.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing `prob` by the name of the function.
        >>> ans = ca1.prob(value)
        >>> print(ans.shape)
        (2,)
        >>> # Evaluate `prob` with respect to distribution b.
        >>> ans = ca1.prob(value, probs_b)
        >>> print(ans.shape)
        (2,)
        >>> # `probs` must be passed in during function calls.
        >>> ans = ca2.prob(value, probs_a)
        >>> print(ans.shape)
        (2,)
        >>> # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     probs (Tensor): event probabilities. Default: self.probs.
        >>> # Examples of `mean`. `sd`, `var`, and `entropy` are similar.
        >>> ans = ca1.mean() # return 0.8
        >>> print(ans.shape)
        (1,)
        >>> ans = ca1.mean(probs_b)
        >>> print(ans.shape)
        (1,)
        >>> # `probs` must be passed in during function calls.
        >>> ans = ca2.mean(probs_a)
        >>> print(ans.shape)
        (1,)
        >>> # Interfaces of `kl_loss` and `cross_entropy` are the same as follows:
        >>> # Args:
        >>> #     dist (str): the name of the distribution. Only 'Categorical' is supported.
        >>> #     probs_b (Tensor): event probabilities of distribution b.
        >>> #     probs (Tensor): event probabilities of distribution a. Default: self.probs.
        >>> # Examples of `kl_loss`, `cross_entropy` is similar.
        >>> ans = ca1.kl_loss('Categorical', probs_b)
        >>> print(ans.shape)
        ()
        >>> ans = ca1.kl_loss('Categorical', probs_b, probs_a)
        >>> print(ans.shape)
        ()
        >>> # An additional `probs` must be passed in.
        >>> ans = ca2.kl_loss('Categorical', probs_b, probs_a)
        >>> print(ans.shape)
        ()
    """

    def __init__(self,
                 probs=None,
                 seed=None,
                 dtype=mstype.int32,
                 name="Categorical"):
        param = dict(locals())
        param['param_dict'] = {'probs': probs}
        valid_dtype = mstype.uint_type + mstype.int_type + mstype.float_type
        Validator.check_type_name(
            "dtype", dtype, valid_dtype, type(self).__name__)
        super(Categorical, self).__init__(seed, dtype, name, param)

        self._probs = self._add_parameter(probs, 'probs')
        if self.probs is not None:
            check_rank(self.probs)
            check_prob(self.probs)
            check_sum_equal_one(probs)

            # update is_scalar_batch and broadcast_shape
            # drop one dimension
            if self.probs.shape[:-1] == ():
                self._is_scalar_batch = True
            self._broadcast_shape = self._broadcast_shape[:-1]

        self.argmax = P.ArgMaxWithValue(axis=-1)
        self.broadcast = broadcast_to
        self.cast = P.Cast()
        self.clip_by_value = ops.clip_by_value
        self.concat = P.Concat(-1)
        self.cumsum = P.CumSum()
        self.dtypeop = P.DType()
        self.exp = exp_generic
        self.expand_dim = P.ExpandDims()
        self.fill = P.Fill()
        self.gather = P.GatherNd()
        self.greater = P.Greater()
        self.issubclass = inner.IsSubClass()
        self.less = P.Less()
        # when the graph kernel mode is enable
        # use Log directly as akg will handle the corner cases
        self.log = P.Log() if context.get_context("enable_graph_kernel") else log_generic
        self.log_softmax = P.LogSoftmax()
        self.logicor = P.LogicalOr()
        self.logicand = P.LogicalAnd()
        self.multinomial = P.Multinomial(seed=self.seed)
        self.reshape = P.Reshape()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.select = P.Select()
        self.shape = P.Shape()
        self.softmax = P.Softmax()
        self.squeeze = P.Squeeze()
        self.squeeze_first_axis = P.Squeeze(0)
        self.squeeze_last_axis = P.Squeeze(-1)
        self.square = P.Square()
        self.transpose = P.Transpose()

        self.index_type = mstype.int32
        self.nan = np.nan

    @property
    def probs(self):
        """
        Return the probability after casting to dtype.

        Output:
            Tensor, the probs of the distribution.
        """
        return self._probs

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            s = 'probs = {}'.format(self.probs)
        else:
            s = 'batch_shape = {}'.format(self._broadcast_shape)
        return s

    def _get_dist_type(self):
        return "Categorical"

    def _get_dist_args(self, probs=None):
        if probs is not None:
            self.checktensor(probs, 'probs')
        else:
            probs = self.probs
        return (probs,)

    def _mean(self, probs=None):
        r"""
        .. math::
            E[X] = \sum_{i=0}^{num_classes-1} i*p_i
        """
        probs = self._check_param_type(probs)
        num_classes = self.shape(probs)[-1]
        index = nn.Range(0., num_classes, 1.)()
        return self.reduce_sum(index * probs, -1)

    def _mode(self, probs=None):
        probs = self._check_param_type(probs)
        index, _ = self.argmax(probs)
        mode = self.cast(index, self.dtype)
        return mode

    def _var(self, probs=None):
        r"""
        .. math::
            VAR(X) = E[X^{2}] - (E[X])^{2}
        """
        probs = self._check_param_type(probs)
        num_classes = self.shape(probs)[-1]
        index = nn.Range(0., num_classes, 1.)()
        return self.reduce_sum(self.square(index) * probs, -1) -\
            self.square(self.reduce_sum(index * probs, -1))

    def _entropy(self, probs=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = -\sum(logits * probs)
        """
        probs = self._check_param_type(probs)
        logits = self.log(probs)
        return self.squeeze(P.Neg()(self.reduce_sum(logits * probs, -1)))

    def _kl_loss(self, dist, probs_b, probs=None):
        """
        Evaluate KL divergence between Categorical distributions.

        Args:
            dist (str): The type of the distributions. Should be "Categorical" in this case.
            probs_b (Tensor): Event probabilities of distribution b.
            probs (Tensor): Event probabilities of distribution a. Default: self.probs.
        """
        check_distribution_name(dist, 'Categorical')
        probs_b = self._check_value(probs_b, 'probs_b')
        probs_b = self.cast(probs_b, self.parameter_type)
        probs_a = self._check_param_type(probs)
        logits_a = self.log(probs_a)
        logits_b = self.log(probs_b)
        return self.squeeze(self.reduce_sum(
            self.softmax(logits_a) * (self.log_softmax(logits_a) - (self.log_softmax(logits_b))), -1))

    def _cross_entropy(self, dist, probs_b, probs=None):
        """
        Evaluate cross entropy between Categorical distributions.

        Args:
            dist (str): The type of the distributions. Should be "Categorical" in this case.
            probs_b (Tensor): Event probabilities of distribution b.
            probs (Tensor): Event probabilities of distribution a. Default: self.probs.
        """
        check_distribution_name(dist, 'Categorical')
        return self._entropy(probs) + self._kl_loss(dist, probs_b, probs)

    def _log_prob(self, value, probs=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            probs (Tensor): Event probabilities. Default: self.probs.
        """
        value = self._check_value(value, 'value')

        probs = self._check_param_type(probs)
        logits = self.log(probs)

        # find the right integer to compute index
        # here we simulate casting to int but still keeping float dtype
        value = self.cast(value, self.dtypeop(probs))

        zeros = self.fill(self.dtypeop(value), self.shape(value), 0.0)
        between_zero_neone = self.logicand(self.less(value, 0,),
                                           self.greater(value, -1.))
        value = self.select(between_zero_neone,
                            zeros,
                            P.Floor()(value))

        # handle the case when value is of shape () and probs is a scalar batch
        drop_dim = False
        if self.shape(value) == () and self.shape(probs)[:-1] == ():
            drop_dim = True
            # manually add one more dimension: () -> (1,)
            # drop this dimension before return
            value = self.expand_dim(value, -1)

        value = self.expand_dim(value, -1)

        broadcast_shape_tensor = logits * value
        broadcast_shape = self.shape(broadcast_shape_tensor)
        num_classes = broadcast_shape[-1]
        label_shape = broadcast_shape[:-1]

        # broadcasting logits and value
        # logit_pmf shape (num of labels, C)
        logits = self.broadcast(logits, broadcast_shape_tensor)
        value = self.broadcast(value, broadcast_shape_tensor)[..., :1]

        # flatten value to shape (number of labels, 1)
        # clip value to be in range from 0 to num_classes -1 and cast into int32
        value = self.reshape(value, (-1, 1))
        out_of_bound = self.squeeze_last_axis(self.logicor(
            self.less(value, 0.0), self.less(num_classes-1, value)))
        # deal with the case the there is only one class.
        value_clipped = self.clip_by_value(value, 0.0, num_classes - 1)
        value_clipped = self.cast(value_clipped, self.index_type)
        # create index from 0 ... NumOfLabels
        index = self.reshape(nn.Range(0, self.shape(value)[0], 1)(), (-1, 1))
        index = self.concat((index, value_clipped))

        # index into logit_pmf, fill in out_of_bound places with -inf
        # reshape into label shape N
        logits_pmf = self.gather(self.reshape(
            logits, (-1, num_classes)), index)
        nan = self.fill(self.dtypeop(logits_pmf),
                        self.shape(logits_pmf), self.nan)
        logits_pmf = self.select(out_of_bound, nan, logits_pmf)
        ans = self.reshape(logits_pmf, label_shape)
        if drop_dim:
            return self.squeeze(ans)
        return ans

    def _cdf(self, value, probs=None):
        r"""
        Cumulative distribution function (cdf) of Categorical distributions.

        Args:
            value (Tensor): The value to be evaluated.
            probs (Tensor): Event probabilities. Default: self.probs.
        """
        value = self._check_value(value, 'value')
        probs = self._check_param_type(probs)

        value = self.cast(value, self.dtypeop(probs))

        zeros = self.fill(self.dtypeop(value), self.shape(value), 0.0)
        between_zero_neone = self.logicand(
            self.less(value, 0,), self.greater(value, -1.))
        value = self.select(between_zero_neone, zeros, P.Floor()(value))

        drop_dim = False
        if self.shape(value) == () and self.shape(probs)[:-1] == ():
            drop_dim = True
            value = self.expand_dim(value, -1)

        value = self.expand_dim(value, -1)

        broadcast_shape_tensor = probs * value
        broadcast_shape = self.shape(broadcast_shape_tensor)
        num_classes = broadcast_shape[-1]
        label_shape = broadcast_shape[:-1]

        probs = self.broadcast(probs, broadcast_shape_tensor)
        value = self.broadcast(value, broadcast_shape_tensor)[..., :1]

        # flatten value to shape (number of labels, 1)
        value = self.reshape(value, (-1, 1))

        # drop one dimension to match cdf
        # clip value to be in range from 0 to num_classes -1 and cast into int32
        less_than_zero = self.squeeze_last_axis(self.less(value, 0.0))
        value_clipped = self.clip_by_value(value, 0.0, num_classes - 1)
        value_clipped = self.cast(value_clipped, self.index_type)

        index = self.reshape(nn.Range(0, self.shape(value)[0], 1)(), (-1, 1))
        index = self.concat((index, value_clipped))

        # reshape probs and fill less_than_zero places with 0
        probs = self.reshape(probs, (-1, num_classes))
        cdf = self.gather(self.cumsum(probs, 1), index)
        zeros = self.fill(self.dtypeop(cdf), self.shape(cdf), 0.0)
        cdf = self.select(less_than_zero, zeros, cdf)
        cdf = self.reshape(cdf, label_shape)

        if drop_dim:
            return self.squeeze(cdf)
        return cdf

    def _sample(self, shape=(), probs=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            probs (Tensor): Event probabilities. Default: self.probs.

        Returns:
            Tensor, shape is shape(probs)[:-1] + sample_shape
        """
        shape = self.checktuple(shape, 'shape')
        probs = self._check_param_type(probs)
        num_classes = self.shape(probs)[-1]
        batch_shape = self.shape(probs)[:-1]

        sample_shape = shape + batch_shape
        drop_dim = False
        if sample_shape == ():
            drop_dim = True
            sample_shape = (1,)

        probs_2d = self.reshape(probs, (-1, num_classes))
        sample_tensor = self.fill(self.dtype, shape, 1.0)
        sample_tensor = self.reshape(sample_tensor, (-1, 1))
        num_sample = self.shape(sample_tensor)[0]
        samples = C.multinomial(probs_2d, num_sample, seed=self.seed)
        samples = self.squeeze(self.transpose(samples, (1, 0)))
        samples = self.cast(self.reshape(samples, sample_shape), self.dtype)
        if drop_dim:
            return self.squeeze_first_axis(samples)
        samples = stop_gradient(samples)
        return samples
