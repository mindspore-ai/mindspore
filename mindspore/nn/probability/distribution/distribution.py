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
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from ._utils.utils import calc_broadcast_shape_from_param, check_scalar_from_param, cast_type_for_device
from ._utils.utils import CheckTuple, CheckTensor


class Distribution(Cell):
    """
    Base class for all mathematical distributions.

    Args:
        seed (int): random seed used in sampling.
        dtype (mindspore.dtype): type of the distribution.
        name (str): Python str name prefixed to Ops created by this class. Default: subclass name.
        param (dict): parameters used to initialize the distribution.

    Note:
        Derived class should override operations such as ,_mean, _prob,
        and _log_prob. Required arguments, such as value for _prob,
        should be passed in through args or kwargs. dist_spec_args which specify
        a new distribution are optional.

        dist_spec_args are unique for each type of distribution. For example, mean and sd
        are the dist_spec_args for a Normal distribution, while rate is the dist_spec_args
        for exponential distribution.

        For all functions, passing in dist_spec_args, is optional.
        Passing in the additional dist_spec_args will make the result to be evaluated with
        new distribution specified by the dist_spec_args. But it won't change the
        original distribuion.
    """

    def __init__(self,
                 seed,
                 dtype,
                 name,
                 param):
        """
        Constructor of distribution class.
        """
        super(Distribution, self).__init__()
        validator.check_value_type('name', name, [str], type(self).__name__)
        validator.check_integer('seed', seed, 0, Rel.GE, name)

        self._name = name
        self._seed = seed
        self._dtype = cast_type_for_device(dtype)
        self._parameters = {}
        # parsing parameters
        for k in param.keys():
            if not(k == 'self' or k.startswith('_')):
                self._parameters[k] = param[k]
        # some attributes
        self._broadcast_shape = calc_broadcast_shape_from_param(
            self.parameters)
        self._is_scalar_batch = check_scalar_from_param(self.parameters)

        # set the function to call according to the derived class's attributes
        self._set_prob()
        self._set_log_prob()
        self._set_sd()
        self._set_var()
        self._set_cdf()
        self._set_survival()
        self._set_log_cdf()
        self._set_log_survival()
        self._set_cross_entropy()

        self.context_mode = context.get_context('mode')
        self.checktuple = CheckTuple()
        self.checktensor = CheckTensor()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def seed(self):
        return self._seed

    @property
    def parameters(self):
        return self._parameters

    @property
    def is_scalar_batch(self):
        return self._is_scalar_batch

    @property
    def broadcast_shape(self):
        return self._broadcast_shape

    def _check_value(self, value, name):
        """
        Check availability fo value as a Tensor.
        """
        if self.context_mode == 0:
            self.checktensor(value, name)
            return value
        return self.checktensor(value, name)

    def _set_prob(self):
        """
        Set probability funtion based on the availability of _prob and _log_likehood.
        """
        if hasattr(self, '_prob'):
            self._call_prob = self._prob
        elif hasattr(self, '_log_prob'):
            self._call_prob = self._calc_prob_from_log_prob

    def _set_sd(self):
        """
        Set standard deviation based on the availability of _sd and _var.
        """
        if hasattr(self, '_sd'):
            self._call_sd = self._sd
        elif hasattr(self, '_var'):
            self._call_sd = self._calc_sd_from_var

    def _set_var(self):
        """
        Set variance based on the availability of _sd and _var.
        """
        if hasattr(self, '_var'):
            self._call_var = self._var
        elif hasattr(self, '_sd'):
            self._call_var = self._calc_var_from_sd

    def _set_log_prob(self):
        """
        Set log probability based on the availability of _prob and _log_prob.
        """
        if hasattr(self, '_log_prob'):
            self._call_log_prob = self._log_prob
        elif hasattr(self, '_prob'):
            self._call_log_prob = self._calc_log_prob_from_prob

    def _set_cdf(self):
        """
        Set cdf based on the availability of _cdf and _log_cdf and survival_functions.
        """
        if hasattr(self, '_cdf'):
            self._call_cdf = self._cdf
        elif hasattr(self, '_log_cdf'):
            self._call_cdf = self._calc_cdf_from_log_cdf
        elif hasattr(self, '_survival_function'):
            self._call_cdf = self._calc_cdf_from_survival
        elif hasattr(self, '_log_survival'):
            self._call_cdf = self._calc_cdf_from_log_survival

    def _set_survival(self):
        """
        Set survival function based on the availability of _survival function and _log_survival
        and _call_cdf.
        """
        if hasattr(self, '_survival_function'):
            self._call_survival = self._survival_function
        elif hasattr(self, '_log_survival'):
            self._call_survival = self._calc_survival_from_log_survival
        elif hasattr(self, '_call_cdf'):
            self._call_survival = self._calc_survival_from_call_cdf

    def _set_log_cdf(self):
        """
        Set log cdf based on the availability of _log_cdf and _call_cdf.
        """
        if hasattr(self, '_log_cdf'):
            self._call_log_cdf = self._log_cdf
        elif hasattr(self, '_call_cdf'):
            self._call_log_cdf = self._calc_log_cdf_from_call_cdf

    def _set_log_survival(self):
        """
        Set log survival based on the availability of _log_survival and _call_survival.
        """
        if hasattr(self, '_log_survival'):
            self._call_log_survival = self._log_survival
        elif hasattr(self, '_call_survival'):
            self._call_log_survival = self._calc_log_survival_from_call_survival

    def _set_cross_entropy(self):
        """
        Set log survival based on the availability of _cross_entropy.
        """
        if hasattr(self, '_cross_entropy'):
            self._call_cross_entropy = self._cross_entropy

    def log_prob(self, *args, **kwargs):
        """
        Evaluate the log probability(pdf or pmf) at the given value.

        Note:
            Args must include value.
            dist_spec_args are optional.
        """
        return self._call_log_prob(*args, **kwargs)

    def _calc_prob_from_log_prob(self, *args, **kwargs):
        r"""
        Evaluate prob from log probability.

        .. math::
            probability(x) = \exp(log_likehood(x))
        """
        return self.exp(self._log_prob(*args, **kwargs))

    def prob(self, *args, **kwargs):
        """
        Evaluate the probability (pdf or pmf) at given value.

        Note:
            Args must include value.
            dist_spec_args are optional.
        """
        return self._call_prob(*args, **kwargs)

    def _calc_log_prob_from_prob(self, *args, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            log_prob(x) = \log(prob(x))
        """
        return self.log(self._prob(*args, **kwargs))

    def cdf(self, *args, **kwargs):
        """
        Evaluate the cdf at given value.

        Note:
            Args must include value.
            dist_spec_args are optional.
        """
        return self._call_cdf(*args, **kwargs)

    def _calc_cdf_from_log_cdf(self, *args, **kwargs):
        r"""
        Evaluate cdf from log_cdf.

        .. math::
            cdf(x) = \exp(log_cdf(x))
        """
        return self.exp(self._log_cdf(*args, **kwargs))

    def _calc_cdf_from_survival(self, *args, **kwargs):
        r"""
        Evaluate cdf from survival function.

        .. math::
            cdf(x) =  1 - (survival_function(x))
        """
        return 1.0 - self._survival_function(*args, **kwargs)

    def _calc_cdf_from_log_survival(self, *args, **kwargs):
        r"""
        Evaluate cdf from log survival function.

        .. math::
            cdf(x) =  1 - (\exp(log_survival(x)))
        """
        return 1.0 - self.exp(self._log_survival(*args, **kwargs))

    def log_cdf(self, *args, **kwargs):
        """
        Evaluate the log cdf at given value.

        Note:
            Args must include value.
            dist_spec_args are optional.
        """
        return self._call_log_cdf(*args, **kwargs)

    def _calc_log_cdf_from_call_cdf(self, *args, **kwargs):
        r"""
        Evaluate log cdf from cdf.

        .. math::
            log_cdf(x) = \log(cdf(x))
        """
        return self.log(self._call_cdf(*args, **kwargs))

    def survival_function(self, *args, **kwargs):
        """
        Evaluate the survival function at given value.

        Note:
            Args must include value.
            dist_spec_args are optional.
        """
        return self._call_survival(*args, **kwargs)

    def _calc_survival_from_call_cdf(self, *args, **kwargs):
        r"""
        Evaluate survival function from cdf.

        .. math::
            survival_function(x) =  1 - (cdf(x))
        """
        return 1.0 - self._call_cdf(*args, **kwargs)

    def _calc_survival_from_log_survival(self, *args, **kwargs):
        r"""
        Evaluate survival function from log survival function.

        .. math::
            survival(x) = \exp(survival_function(x))
        """
        return self.exp(self._log_survival(*args, **kwargs))

    def log_survival(self, *args, **kwargs):
        """
        Evaluate the log survival function at given value.

        Note:
            Args must include value.
            dist_spec_args are optional.
        """
        return self._call_log_survival(*args, **kwargs)

    def _calc_log_survival_from_call_survival(self, *args, **kwargs):
        r"""
        Evaluate log survival function from survival function.

        .. math::
            log_survival(x) = \log(survival_function(x))
        """
        return self.log(self._call_survival(*args, **kwargs))

    def kl_loss(self, *args, **kwargs):
        """
        Evaluate the KL divergence, i.e. KL(a||b).

        Note:
            Args must include type of the distribution, parameters of distribution b.
            Parameters for distribution a are optional.
        """
        return self._kl_loss(*args, **kwargs)

    def mean(self, *args, **kwargs):
        """
        Evaluate the mean.

        Note:
            dist_spec_args are optional.
        """
        return self._mean(*args, **kwargs)

    def mode(self, *args, **kwargs):
        """
        Evaluate the mode.

        Note:
            dist_spec_args are optional.
        """
        return self._mode(*args, **kwargs)

    def sd(self, *args, **kwargs):
        """
        Evaluate the standard deviation.

        Note:
            dist_spec_args are optional.
        """
        return self._call_sd(*args, **kwargs)

    def var(self, *args, **kwargs):
        """
        Evaluate the variance.

        Note:
            dist_spec_args are optional.
        """
        return self._call_var(*args, **kwargs)

    def _calc_sd_from_var(self, *args, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            STD(x) = \sqrt(VAR(x))
        """
        return self.sqrt(self._var(*args, **kwargs))

    def _calc_var_from_sd(self, *args, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            VAR(x) = STD(x) ^ 2
        """
        return self.sq(self._sd(*args, **kwargs))

    def entropy(self, *args, **kwargs):
        """
        Evaluate the entropy.

        Note:
            dist_spec_args are optional.
        """
        return self._entropy(*args, **kwargs)

    def cross_entropy(self, *args, **kwargs):
        """
        Evaluate the cross_entropy between distribution a and b.

        Note:
            Args must include type of the distribution, parameters of distribution b.
            Parameters for distribution a are optional.
        """
        return self._call_cross_entropy(*args, **kwargs)

    def _calc_cross_entropy(self, *args, **kwargs):
        r"""
        Evaluate cross_entropy from entropy and kl divergence.

        .. math::
            H(X, Y) = H(X) + KL(X||Y)
        """
        return self._entropy(*args, **kwargs) + self._kl_loss(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Sampling function.

        Note:
            Shape of the sample is default to ().
            dist_spec_args are optional.
        """
        return self._sample(*args, **kwargs)

    def construct(self, name, *args, **kwargs):
        """
        Override construct in Cell.

        Note:
            Names of supported functions include:
            'prob', 'log_prob', 'cdf', 'log_cdf', 'survival_function', 'log_survival'
            'var', 'sd', 'entropy', 'kl_loss', 'cross_entropy', 'sample'.

        Args:
            name (str): name of the function.
            *args (list): list of positional arguments needed for the function.
            **kwargs (dictionary): dictionary of keyword arguments needed for the function.
        """

        if name == 'log_prob':
            return self._call_log_prob(*args, **kwargs)
        if name == 'prob':
            return self._call_prob(*args, **kwargs)
        if name == 'cdf':
            return self._call_cdf(*args, **kwargs)
        if name == 'log_cdf':
            return self._call_log_cdf(*args, **kwargs)
        if name == 'survival_function':
            return self._call_survival(*args, **kwargs)
        if name == 'log_survival':
            return self._call_log_survival(*args, **kwargs)
        if name == 'kl_loss':
            return self._kl_loss(*args, **kwargs)
        if name == 'mean':
            return self._mean(*args, **kwargs)
        if name == 'mode':
            return self._mode(*args, **kwargs)
        if name == 'sd':
            return self._call_sd(*args, **kwargs)
        if name == 'var':
            return self._call_var(*args, **kwargs)
        if name == 'entropy':
            return self._entropy(*args, **kwargs)
        if name == 'cross_entropy':
            return self._call_cross_entropy(*args, **kwargs)
        if name == 'sample':
            return self._sample(*args, **kwargs)
        return None
