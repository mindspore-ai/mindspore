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
"""basic"""
from ..cell import Cell
from ._utils.utils import calc_broadcast_shape_from_param


class Distribution(Cell):
    """
    Base class for all mathematical distributions.

    Note:
        Derived class should override operations such as ,_mean, _prob,
        and _log_prob. Functions should be called through construct when
        used inside a network in the form  of function name followed by
        arguments.

    Examples:
        >>> class MyNormalDistribution(Distribution):
        >>>    def __init__(self):
        >>>        super(MyDistribution, self).__init__()
        >>>        self._mean_value = Tensor([2.0,3.0])
        >>>        self._sd_value = Tensor([2.0,3.0])
        >>>
        >>>    def _mean(self):
        >>>        return self._mean_value

    """
    def __init__(self,
                 dtype,
                 name,
                 param):

        """
        Constructor of distribution class.
        """
        super(Distribution, self).__init__()
        self._name = name
        self._dtype = dtype
        self._parameters = {}
        # parsing parameters
        for k in param.keys():
            if not(k == 'self' or k.startswith('_')):
                self._parameters[k] = param[k]
        # some attributes
        self._broadcast_shape = calc_broadcast_shape_from_param(
            self._parameters)

        # set the function to call according to the derived class's attributes
        self._set_prob()
        self._set_log_prob()
        self._set_sd()

    def _set_prob(self):
        """
        Set probability funtion based on the availability of _prob and _log_likehood.
        """
        if hasattr(self, '_prob'):
            self._call_prob = self._prob
        elif hasattr(self, '_log_likelihood'):
            self._call_prob = self._calc_prob_from_log_likelihood

    def _set_sd(self):
        """
        Set standard deviation based on the availability of _sd and _var.
        """
        if hasattr(self, '_sd'):
            self._call_sd = self._sd
        elif hasattr(self, '_var'):
            self._call_sd = self._calc_sd_from_var

    def _set_log_prob(self):
        """
        Set log probability based on the availability of _prob and _log_likelihood.
        """
        if hasattr(self, '_log_likelihood'):
            self._call_log_prob = self._log_likelihood
        if hasattr(self, '_prob'):
            self._call_log_prob = self._calc_log_prob_from_prob

    def log_likelihood(self, *args):
        """
        Evaluate the log probability at the given value.

        Note:
            value is casted to Tensor for further calculation.

        Args:
            name (str): name of the calling function.
            value (Tensor): values to be evaluated.
            mean (Tensor): mean of the distirbution. Default: self.mean.
            sd (Tensor): standard deviation of the distribution. Default: self.sd.

        Outputs:
            Tensor, shape: broadcast_shape of the distribution.
        """
        return self._call_log_prob(*args)

    def _calc_prob_from_log_likelihood(self, *args):
        r"""
        Evaluate prob from log probability.

        .. math::
            probability(x) = \exp(log_likehood(x))

        Args:
            name (str): name of the calling function.
            value (Tensor): values to be evaluated.
            mean (Tensor): mean of the distribution. Default: self.mean.
            sd (Tensor): standard deviation of the distritbuion. Default: self.sd.
        """
        return self.exp(self._log_likelihood(*args))

    def _call_prob(self, *args):
        """
        Raises:
            NotImplementedError when derived class didn't override _prob or _log_likelihood.
        """
        raise NotImplementedError('pdf/pmf is not implemented: {}'.format(type(self).__name__))

    def _call_log_prob(self, *args):
        """
        Raises:
            NotImplementedError when derived class didn't override _prob or _log_likelihood.
        """
        raise NotImplementedError('log_probability is not implemented: {}'.format(type(self).__name__))

    def _call_sd(self):
        """
        Raises:
            NotImplementedError when derived class didn't override _sd or _var.
        """
        raise NotImplementedError('standard deviation is not implemented: {}'.format(type(self).__name__))

    def prob(self, *args):
        """
        Evaluate the prob (pdf or pmf) at given value.

        Note:
            value is casted to Tensor for further calculation.

        Args:
            name (str): name of the calling function.
            value (Tensor): values to be evaluated.
            mean (Tensor): mean of the distribution.
            sd (Tensor): standard deviation of the distritbuion.

        Outputs:
            Tensor, shape: broadcast_shape of the distribution.
        """
        return self._call_prob(*args)

    def _calc_log_prob_from_prob(self, *args):
        r"""
        Evaluate log probability from probability.

        .. math::
            log_prob(x) = \log(prob(x))
        """
        return self.log(self._prob(*args))

    def kl_loss(self, **kwargs):
        """
        Evaluate the KL divergence. Parameters of the second distribution should be
        passed in through **kwargs.

        Outputs:
            Tensor, shape: broadcast_shape of the distribution and input distribution.
        """
        return self._kl_loss(**kwargs)

    def mean(self, **kwargs):
        """
        Evaluate the mean.

        Outputs:
            Tensor, shape: broadcast_shape of the distribution.
        """
        return self._mean(**kwargs)

    def sd(self, **kwargs):
        """
        Evaluate the standard deviation.

        Outputs:
            Tensor, with shape of broadcast_shape of the distribution.
        """
        return self._call_sd(**kwargs)

    def _calc_sd_from_var(self, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            STD(x) = \sqrt(VAR(x))
        """
        return self.sqrt(self._var(**kwargs))

    def construct(self, *inputs):
        """
        Override construct in Cell.

        Args:
            *inputs: inputs[0] is always the name of the function.

        Notes:
            Always raise RuntimeError as Distribution should not be called directly.
        """

        if inputs[0] == 'log_prob':
            return self._call_log_prob(*inputs)
        if inputs[0] == 'prob':
            return self._call_prob(*inputs)
        if inputs[0] == 'kl_loss':
            return self._kl_loss(*inputs)
        if inputs[0] == 'mean':
            return self._mean()
        if inputs[0] == 'sd':
            return self._call_sd()
        return None
