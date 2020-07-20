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
from ._utils.utils import calc_broadcast_shape_from_param, check_scalar_from_param

class Distribution(Cell):
    """
    Base class for all mathematical distributions.

    Args:
        dtype (mindspore.dtype): type of the distribution.
        name (str): name of the distribution.
        param (dict): parameters used to initialize the distribution.

    Note:
        Derived class should override operations such as ,_mean, _prob,
        and _log_prob. Functions should be called through construct when
        used inside a network. Arguments should be passed in through *args
        in the form  of function name followed by additional arguments.
        Functions such as cdf and prob, require a value to be passed in while
        functions such as mean and sd do not require arguments other than name.

        Dist_spec_args are unique for each type of distribution. For example, mean and sd
        are the dist_spec_args for a Normal distribution.

        For all functions, passing in dist_spec_args, are optional.
        Passing in the additional dist_spec_args will make the result to be evaluated with
        new distribution specified by the dist_spec_args. But it won't change the
        original distribuion.
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

        self._prob_functions = ('prob', 'log_prob')
        self._cdf_survival_functions = ('cdf', 'log_cdf', 'survival_function', 'log_survival')
        self._variance_functions = ('var', 'sd')
        self._divergence_functions = ('kl_loss', 'cross_entropy')

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def parameters(self):
        return self._parameters

    @property
    def is_scalar_batch(self):
        return self._is_scalar_batch

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

    def log_prob(self, *args):
        """
        Evaluate the log probability(pdf or pmf) at the given value.

        Note:
            Args must include name of the function and value.
            Dist_spec_args are optional.
        """
        return self._call_log_prob(*args)

    def _calc_prob_from_log_prob(self, *args):
        r"""
        Evaluate prob from log probability.

        .. math::
            probability(x) = \exp(log_likehood(x))
        """
        return self.exp(self._log_prob(*args))

    def prob(self, *args):
        """
        Evaluate the probability (pdf or pmf) at given value.

        Note:
            Args must include name of the function and value.
            Dist_spec_args are optional.
        """
        return self._call_prob(*args)

    def _calc_log_prob_from_prob(self, *args):
        r"""
        Evaluate log probability from probability.

        .. math::
            log_prob(x) = \log(prob(x))
        """
        return self.log(self._prob(*args))

    def cdf(self, *args):
        """
        Evaluate the cdf at given value.

        Note:
            Args must include name of the function and value.
            Dist_spec_args are optional.
        """
        return self._call_cdf(*args)

    def _calc_cdf_from_log_cdf(self, *args):
        r"""
        Evaluate cdf from log_cdf.

        .. math::
            cdf(x) = \exp(log_cdf(x))
        """
        return self.exp(self._log_cdf(*args))

    def _calc_cdf_from_survival(self, *args):
        r"""
        Evaluate cdf from survival function.

        .. math::
            cdf(x) =  1 - (survival_function(x))
        """
        return 1.0 - self._survival_function(*args)

    def _calc_cdf_from_log_survival(self, *args):
        r"""
        Evaluate cdf from log survival function.

        .. math::
            cdf(x) =  1 - (\exp(log_survival(x)))
        """
        return 1.0 - self.exp(self._log_survival(*args))

    def log_cdf(self, *args):
        """
        Evaluate the log cdf at given value.

        Note:
            Args must include name of the function and value.
            Dist_spec_args are optional.
        """
        return self._call_log_cdf(*args)

    def _calc_log_cdf_from_call_cdf(self, *args):
        r"""
        Evaluate log cdf from cdf.

        .. math::
            log_cdf(x) = \log(cdf(x))
        """
        return self.log(self._call_cdf(*args))

    def survival_function(self, *args):
        """
        Evaluate the survival function at given value.

        Note:
            Args must include name of the function and value.
            Dist_spec_args are optional.
        """
        return self._call_survival(*args)

    def _calc_survival_from_call_cdf(self, *args):
        r"""
        Evaluate survival function from cdf.

        .. math::
            survival_function(x) =  1 - (cdf(x))
        """
        return 1.0 - self._call_cdf(*args)

    def _calc_survival_from_log_survival(self, *args):
        r"""
        Evaluate survival function from log survival function.

        .. math::
            survival(x) = \exp(survival_function(x))
        """
        return self.exp(self._log_survival(*args))

    def log_survival(self, *args):
        """
        Evaluate the log survival function at given value.

        Note:
            Args must include name of the function and value.
            Dist_spec_args are optional.
        """
        return self._call_log_survival(*args)

    def _calc_log_survival_from_call_survival(self, *args):
        r"""
        Evaluate log survival function from survival function.

        .. math::
            log_survival(x) = \log(survival_function(x))
        """
        return self.log(self._call_survival(*args))

    def kl_loss(self, *args):
        """
        Evaluate the KL divergence, i.e. KL(a||b).

        Note:
            Args must include name of the function, type of the distribution, parameters of distribution b.
            Parameters for distribution a are optional.
        """
        return self._kl_loss(*args)

    def mean(self, *args):
        """
        Evaluate the mean.

        Note:
            Args must include the name of function. Dist_spec_args are optional.
        """
        return self._mean(*args)

    def mode(self, *args):
        """
        Evaluate the mode.

        Note:
            Args must include the name of function. Dist_spec_args are optional.
        """
        return self._mode(*args)

    def sd(self, *args):
        """
        Evaluate the standard deviation.

        Note:
            Args must include the name of function. Dist_spec_args are optional.
        """
        return self._call_sd(*args)

    def var(self, *args):
        """
        Evaluate the variance.

        Note:
            Args must include the name of function. Dist_spec_args are optional.
        """
        return self._call_var(*args)

    def _calc_sd_from_var(self, *args):
        r"""
        Evaluate log probability from probability.

        .. math::
            STD(x) = \sqrt(VAR(x))
        """
        return self.sqrt(self._var(*args))

    def _calc_var_from_sd(self, *args):
        r"""
        Evaluate log probability from probability.

        .. math::
            VAR(x) = STD(x) ^ 2
        """
        return self.sq(self._sd(*args))

    def entropy(self, *args):
        """
        Evaluate the entropy.

        Note:
            Args must include the name of function. Dist_spec_args are optional.
        """
        return self._entropy(*args)

    def cross_entropy(self, *args):
        """
        Evaluate the cross_entropy between distribution a and b.

        Note:
            Args must include name of the function, type of the distribution, parameters of distribution b.
            Parameters for distribution a are optional.
        """
        return self._call_cross_entropy(*args)

    def _calc_cross_entropy(self, *args):
        r"""
        Evaluate cross_entropy from entropy and kl divergence.

        .. math::
            H(X, Y) = H(X) + KL(X||Y)
        """
        return self._entropy(*args) + self._kl_loss(*args)

    def sample(self, *args):
        """
        Sampling function.

        Args:
            *args (list): arguments passed in through construct.

        Note:
            Args must include name of the function.
            Shape of the sample and dist_spec_args are optional.
        """
        return self._sample(*args)


    def construct(self, *inputs):
        """
        Override construct in Cell.

        Note:
            Names of supported functions:
            'prob', 'log_prob', 'cdf', 'log_cdf', 'survival_function', 'log_survival'
            'var', 'sd', 'entropy', 'kl_loss', 'cross_entropy', 'sample'.

        Args:
            *inputs (list): inputs[0] is always the name of the function.
        """

        if inputs[0] == 'log_prob':
            return self._call_log_prob(*inputs)
        if inputs[0] == 'prob':
            return self._call_prob(*inputs)
        if inputs[0] == 'cdf':
            return self._call_cdf(*inputs)
        if inputs[0] == 'log_cdf':
            return self._call_log_cdf(*inputs)
        if inputs[0] == 'survival_function':
            return self._call_survival(*inputs)
        if inputs[0] == 'log_survival':
            return self._call_log_survival(*inputs)
        if inputs[0] == 'kl_loss':
            return self._kl_loss(*inputs)
        if inputs[0] == 'mean':
            return self._mean(*inputs)
        if inputs[0] == 'mode':
            return self._mode(*inputs)
        if inputs[0] == 'sd':
            return self._call_sd(*inputs)
        if inputs[0] == 'var':
            return self._call_var(*inputs)
        if inputs[0] == 'entropy':
            return self._entropy(*inputs)
        if inputs[0] == 'cross_entropy':
            return self._call_cross_entropy(*inputs)
        if inputs[0] == 'sample':
            return self._sample(*inputs)
        return None
