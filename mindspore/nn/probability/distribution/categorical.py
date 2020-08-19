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
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import logits_to_probs, probs_to_logits, check_tensor_type, cast_to_tensor


class Categorical(Distribution):
    """
    Creates a categorical distribution parameterized by either probs or logits (but not both).

    Args:
        probs (Tensor, list, numpy.ndarray, Parameter, float): event probabilities.
        logits (Tensor, list, numpy.ndarray, Parameter, float): event log-odds.
        seed (int): seed to use in sampling. Default: 0.
        dtype (mindspore.dtype): type of the distribution. Default: mstype.int32.
        name (str): name of the distribution. Default: Categorical.

    Note:
        probs must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1.

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
        >>>         # value should be Tensor
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
                 seed=0,
                 dtype=mstype.int32,
                 name="Categorical"):
        param = dict(locals())
        super(Categorical, self).__init__(seed, dtype, name, param)
        if (probs is None) == (logits is None):
            raise ValueError("Either 'prob' or 'logits' must be specified, but not both.")
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.log = P.Log()
        self.exp = P.Exp()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.div = P.RealDiv()
        self.size = P.Size()
        self.mutinomial = P.Multinomial(seed=seed)
        self.cast = P.Cast()
        self.expandim = P.ExpandDims()
        self.gather = P.GatherNd()
        self.concat = P.Concat(-1)
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
        self._batch_shape = self.shape(self._param2d)[0]


    @property
    def logits(self):
        """
        Returns the logits.
        """
        return self._logits

    @property
    def probs(self):
        """
        Returns the probability.
        """
        return self._probs

    def _sample(self, sample_shape=(1,)):
        """
        Sampling.

        Args:
            sample_shape (tuple): shape of the sample. Default: (1,).

        Returns:
            Tensor, shape is shape(probs)[:-1] + sample_shape
        """
        if not isinstance(sample_shape, tuple):
            raise ValueError("sample shape must be a tuple")
        num_sample = 1
        for i in sample_shape:
            num_sample *= i
        probs_2d = self.reshape(self._probs, (-1, self._num_events))
        samples = self.mutinomial(probs_2d, num_sample)
        extend_shape = sample_shape
        if len(self.shape(self._probs)) > 1:
            extend_shape = self.shape(self._probs)[:-1] + sample_shape
        return self.cast(self.reshape(samples, extend_shape), self.dtype)

    def _broad_cast_shape(self, a, b):
        """
        Broadcast Tensor shape.

        Args:
            a (Tensor): A Tensor need to Broadcast.
            b (Tensor): Another Tensor need to Broadcast.

        Returns:
            Tuple, Broadcast shape.
        """
        shape_a = self.shape(a)
        shape_b = self.shape(b)
        size_a = len(shape_a)
        size_b = len(shape_b)
        if size_a > size_b:
            size = size_a
            shape_out = list(shape_a)
            shape_short = list(shape_b)
            diff_size = size_a - size_b
        else:
            size = size_b
            shape_out = list(shape_b)
            shape_short = list(shape_a)
            diff_size = size_b - size_a
        for i in range(diff_size, size):
            if shape_out[i] == shape_short[i - diff_size]:
                continue
            if shape_out[i] == 1 or shape_short[i - diff_size] == 1:
                shape_out[i] = shape_out[i] * shape_short[i - diff_size]
            else:
                raise ValueError(f"Shape {shape_a} and {shape_b} is not broadcastable.")
        return tuple(shape_out)

    def _log_prob(self, value):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): value to be evaluated. The dtype could be mstype.float32, bool, mstype.int32.
        """
        if value is not None:
            check_tensor_type("value", value, [mstype.float32, bool, mstype.int32])
            value = self.expandim(self.cast(value, mstype.float32), -1)
            broad_shape = self._broad_cast_shape(value, self._logits)
            broad = P.BroadcastTo(broad_shape)
            value = broad(value)[..., :1]
            index = cast_to_tensor(np.arange(broad_shape[-1]).astype(np.float32))
            index = self.expandim(index, -1)
            index = broad(index)[..., :1]
            value = self.concat((index, value))
            value = self.cast(value, mstype.int32)
            return self.gather(self._logits, value)
        return None

    def _entropy(self):
        r"""
       Evaluate entropy.

       .. math::
           H(X) = -\sum(logits * probs)
       """
        p_log_p = self._logits * self._probs
        return self.reduce_sum(-p_log_p, -1)

    def enumerate_support(self, expand=True):
        r"""
       Enumerate categories.
       """
        num_events = self._num_events
        values = cast_to_tensor(np.arange(num_events).astype(np.int32), mstype.int32)
        values = self.reshape(values, (num_events, 1))
        if expand:
            values = P.BroadcastTo((num_events, self._batch_shape))(values)
        values = self.cast(values, mstype.int32)
        return values
