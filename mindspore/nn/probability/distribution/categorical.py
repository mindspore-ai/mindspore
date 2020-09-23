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
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import logits_to_probs, probs_to_logits, check_type, cast_to_tensor, \
    raise_probs_logits_error


class Categorical(Distribution):
    """
    Create a categorical distribution parameterized by either probabilities or logits (but not both).

    Args:
        probs (Tensor, list, numpy.ndarray, Parameter): Event probabilities.
        logits (Tensor, list, numpy.ndarray, Parameter, float): Event log-odds.
        seed (int): The global seed is used in sampling. Global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the distribution. Default: mstype.int32.
        name (str): The name of the distribution. Default: Categorical.

    Note:
        `probs` must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1.

    Examples:
        >>> # To initialize a Categorical distribution of prob is [0.5, 0.5]
        >>> import mindspore.nn.probability.distribution as msd
        >>> b = msd.Categorical(probs = [0.5, 0.5], dtype=mstype.int32)
        >>>
        >>> # To use Categorical in a network
        >>> class net(Cell):
        >>>     def __init__(self, probs):
        >>>         super(net, self).__init__():
        >>>         self.ca = msd.Categorical(probs=probs, dtype=mstype.int32)
        >>>     # All the following calls in construct are valid
        >>>     def construct(self, value):
        >>>
        >>>         # Similar calls can be made to logits
        >>>         ans = self.ca.probs
        >>>         # value must be Tensor(mstype.float32, bool, mstype.int32)
        >>>         ans = self.ca.log_prob(value)
        >>>
        >>>         # Usage of enumerate_support
        >>>         ans = self.ca.enumerate_support()
        >>>
        >>>         # Usage of entropy
        >>>         ans = self.ca.entropy()
        >>>
        >>>         # Sample
        >>>         ans = self.ca.sample()
        >>>         ans = self.ca.sample((2,3))
        >>>         ans = self.ca.sample((2,))
    """

    def __init__(self,
                 probs=None,
                 logits=None,
                 seed=None,
                 dtype=mstype.int32,
                 name="Categorical"):
        param = dict(locals())
        valid_dtype = mstype.int_type
        check_type(dtype, valid_dtype, "Categorical")
        super(Categorical, self).__init__(seed, dtype, name, param)
        if (probs is None) == (logits is None):
            raise_probs_logits_error()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reduce_sum1 = P.ReduceSum(keep_dims=False)
        self.log = P.Log()
        self.exp = P.Exp()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.div = P.RealDiv()
        self.size = P.Size()
        self.mutinomial = P.Multinomial(seed=self.seed)
        self.cast = P.Cast()
        self.expandim = P.ExpandDims()
        self.gather = P.GatherNd()
        self.concat = P.Concat(-1)
        self.transpose = P.Transpose()
        if probs is not None:
            self._probs = cast_to_tensor(probs, mstype.float32)
            input_sum = self.reduce_sum(self._probs, -1)
            self._probs = self.div(self._probs, input_sum)
            self._logits = probs_to_logits(self._probs)
            self._param = self._probs
        else:
            self._logits = cast_to_tensor(logits, mstype.float32)
            input_sum = self.reduce_sum(self.exp(self._logits), -1)
            self._logits = self._logits - self.log(input_sum)
            self._probs = logits_to_probs(self._logits)
            self._param = self._logits
        self._num_events = self.shape(self._param)[-1]
        self._param2d = self.reshape(self._param, (-1, self._num_events))
        self._batch_shape = self.shape(self._param)[:-1]
        self._batch_shape_n = (1,) * len(self._batch_shape)

    @property
    def logits(self):
        """
        Return the logits.
        """
        return self._logits

    @property
    def probs(self):
        """
        Return the probability.
        """
        return self._probs

    def _sample(self, sample_shape=()):
        """
        Sampling.

        Args:
            sample_shape (tuple): The shape of the sample. Default: ().

        Returns:
            Tensor, shape is shape(probs)[:-1] + sample_shape
        """
        self.checktuple(sample_shape, 'shape')
        num_sample = 1
        for i in sample_shape:
            num_sample *= i
        probs_2d = self.reshape(self._probs, (-1, self._num_events))
        samples = self.mutinomial(probs_2d, num_sample)
        samples = self.transpose(samples, (1, 0))
        extend_shape = sample_shape
        if len(self.shape(self._probs)) > 1:
            extend_shape = sample_shape + self.shape(self._probs)[:-1]
        return self.cast(self.reshape(samples, extend_shape), self.dtype)

    def _log_prob(self, value):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
        """
        value = self._check_value(value, 'value')
        value = self.expandim(self.cast(value, mstype.float32), -1)
        broad_shape = self.shape(value + self._logits)
        broad = P.BroadcastTo(broad_shape)
        logits_pmf = self.reshape(broad(self._logits), (-1, broad_shape[-1]))
        value = self.reshape(broad(value)[..., :1], (-1, 1))
        index = nn.Range(0., self.shape(value)[0], 1)()
        index = self.reshape(index, (-1, 1))
        value = self.concat((index, value))
        value = self.cast(value, mstype.int32)
        return self.reshape(self.gather(logits_pmf, value), broad_shape[:-1])

    def _entropy(self):
        r"""
       Evaluate entropy.

       .. math::
           H(X) = -\sum(logits * probs)
       """
        p_log_p = self._logits * self._probs
        return self.reduce_sum1(-p_log_p, -1)

    def enumerate_support(self, expand=True):
        r"""
       Enumerate categories.

       Args:
           expand (Bool): Whether to expand.
       """
        num_events = self._num_events
        values = nn.Range(0., num_events, 1)()
        values = self.reshape(values, (num_events,) + self._batch_shape_n)
        if expand:
            values = P.BroadcastTo((num_events,) + self._batch_shape)(values)
        values = self.cast(values, mstype.int32)
        return values
