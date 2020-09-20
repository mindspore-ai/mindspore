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
from mindspore.common import get_seed
from ._utils.utils import calc_broadcast_shape_from_param, check_scalar_from_param, cast_type_for_device,\
    raise_none_error
from ._utils.utils import CheckTuple, CheckTensor


class Distribution(Cell):
    """
    Base class for all mathematical distributions.

    Args:
        seed (int): The seed is used in sampling. The global seed is used if it is None.
        dtype (mindspore.dtype): The type of the event samples.
        name (str): The name of the distribution.
        param (dict): The parameters used to initialize the distribution.

    Note:
        Derived class must override operations such as `_mean`, `_prob`,
        and `_log_prob`. Required arguments, such as `value` for `_prob`,
        must be passed in through `args` or `kwargs`. `dist_spec_args` which specifies
        a new distribution are optional.

        `dist_spec_args` is unique for each type of distribution. For example, `mean` and `sd`
        are the `dist_spec_args` for a Normal distribution, while `rate` is the `dist_spec_args`
        for an Exponential distribution.

        For all functions, passing in `dist_spec_args`, is optional.
        Function calls with the additional `dist_spec_args` passed in will evaluate the result with
        a new distribution specified by the `dist_spec_args`. However, it will not change the original distribution.
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
        if seed is None:
            seed = get_seed()
            if seed is None:
                seed = 0
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

    def _check_param_type(self, *args):
        """
        Check the availability and validity of default parameters and `dist_spec_args`.
        `dist_spec_args` passed in must be tensors. If default parameters of the distribution
        are None, the parameters must be passed in through `args`.
        """
        broadcast_shape = None
        common_dtype = None
        out = []

        for arg, name, default in zip(args, self.parameter_names, self.default_parameters):
            # check if the argument is a Tensor
            if arg is not None:
                if self.context_mode == 0:
                    self.checktensor(arg, name)
                else:
                    arg = self.checktensor(arg, name)
            else:
                arg = default if default is not None else raise_none_error(
                    name)

            # broadcast if the number of args > 1
            if broadcast_shape is None:
                broadcast_shape = self.shape(arg)
                common_dtype = self.dtypeop(arg)
            else:
                ones = self.fill(self.dtypeop(arg), broadcast_shape, 1.0)
                broadcast_shape = self.shape(arg + ones)

                # check if the arguments have the same dtype
                arg = arg * self.fill(self.dtypeop(arg), broadcast_shape, 1.0)
                dtype_tensor = self.fill(common_dtype, broadcast_shape, 1.0)
                self.sametypeshape(arg, dtype_tensor)
            arg = self.cast(arg, self.parameter_type)
            out.append(arg)

        if len(out) == 1:
            return out[0]

        # broadcast all args to broadcast_shape
        result = ()
        for arg in out:
            arg = arg * self.fill(self.dtypeop(arg), broadcast_shape, 1.0)
            result = result + (arg,)
        return result

    def _check_value(self, value, name):
        """
        Check availability of `value` as a Tensor.
        """
        if self.context_mode == 0:
            self.checktensor(value, name)
            return value
        return self.checktensor(value, name)

    def _set_prob(self):
        """
        Set probability funtion based on the availability of `_prob` and `_log_likehood`.
        """
        if hasattr(self, '_prob'):
            self._call_prob = self._prob
        elif hasattr(self, '_log_prob'):
            self._call_prob = self._calc_prob_from_log_prob

    def _set_sd(self):
        """
        Set standard deviation based on the availability of `_sd` and `_var`.
        """
        if hasattr(self, '_sd'):
            self._call_sd = self._sd
        elif hasattr(self, '_var'):
            self._call_sd = self._calc_sd_from_var

    def _set_var(self):
        """
        Set variance based on the availability of `_sd` and `_var`.
        """
        if hasattr(self, '_var'):
            self._call_var = self._var
        elif hasattr(self, '_sd'):
            self._call_var = self._calc_var_from_sd

    def _set_log_prob(self):
        """
        Set log probability based on the availability of `_prob` and `_log_prob`.
        """
        if hasattr(self, '_log_prob'):
            self._call_log_prob = self._log_prob
        elif hasattr(self, '_prob'):
            self._call_log_prob = self._calc_log_prob_from_prob

    def _set_cdf(self):
        """
        Set cumulative distribution function (cdf) based on the availability of `_cdf` and `_log_cdf` and
        `survival_functions`.
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
        Set survival function based on the availability of _survival function and `_log_survival`
        and `_call_cdf`.
        """
        if hasattr(self, '_survival_function'):
            self._call_survival = self._survival_function
        elif hasattr(self, '_log_survival'):
            self._call_survival = self._calc_survival_from_log_survival
        elif hasattr(self, '_call_cdf'):
            self._call_survival = self._calc_survival_from_call_cdf

    def _set_log_cdf(self):
        """
        Set log cdf based on the availability of `_log_cdf` and `_call_cdf`.
        """
        if hasattr(self, '_log_cdf'):
            self._call_log_cdf = self._log_cdf
        elif hasattr(self, '_call_cdf'):
            self._call_log_cdf = self._calc_log_cdf_from_call_cdf

    def _set_log_survival(self):
        """
        Set log survival based on the availability of `_log_survival` and `_call_survival`.
        """
        if hasattr(self, '_log_survival'):
            self._call_log_survival = self._log_survival
        elif hasattr(self, '_call_survival'):
            self._call_log_survival = self._calc_log_survival_from_call_survival

    def _set_cross_entropy(self):
        """
        Set log survival based on the availability of `_cross_entropy`.
        """
        if hasattr(self, '_cross_entropy'):
            self._call_cross_entropy = self._cross_entropy

    def log_prob(self, value, *args, **kwargs):
        """
        Evaluate the log probability(pdf or pmf) at the given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its dist_spec_args through
            `args` or `kwargs`.
        """
        return self._call_log_prob(value, *args, **kwargs)

    def _calc_prob_from_log_prob(self, value, *args, **kwargs):
        r"""
        Evaluate prob from log probability.

        .. math::
            probability(x) = \exp(log_likehood(x))
        """
        return self.exp(self._log_prob(value, *args, **kwargs))

    def prob(self, value, *args, **kwargs):
        """
        Evaluate the probability (pdf or pmf) at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its dist_spec_args through
            `args` or `kwargs`.
        """
        return self._call_prob(value, *args, **kwargs)

    def _calc_log_prob_from_prob(self, value, *args, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            log_prob(x) = \log(prob(x))
        """
        return self.log(self._prob(value, *args, **kwargs))

    def cdf(self, value, *args, **kwargs):
        """
        Evaluate the cdf at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its dist_spec_args through
            `args` or `kwargs`.
        """
        return self._call_cdf(value, *args, **kwargs)

    def _calc_cdf_from_log_cdf(self, value, *args, **kwargs):
        r"""
        Evaluate cdf from log_cdf.

        .. math::
            cdf(x) = \exp(log_cdf(x))
        """
        return self.exp(self._log_cdf(value, *args, **kwargs))

    def _calc_cdf_from_survival(self, value, *args, **kwargs):
        r"""
        Evaluate cdf from survival function.

        .. math::
            cdf(x) =  1 - (survival_function(x))
        """
        return 1.0 - self._survival_function(value, *args, **kwargs)

    def _calc_cdf_from_log_survival(self, value, *args, **kwargs):
        r"""
        Evaluate cdf from log survival function.

        .. math::
            cdf(x) =  1 - (\exp(log_survival(x)))
        """
        return 1.0 - self.exp(self._log_survival(value, *args, **kwargs))

    def log_cdf(self, value, *args, **kwargs):
        """
        Evaluate the log cdf at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its dist_spec_args through
            `args` or `kwargs`.
        """
        return self._call_log_cdf(value, *args, **kwargs)

    def _calc_log_cdf_from_call_cdf(self, value, *args, **kwargs):
        r"""
        Evaluate log cdf from cdf.

        .. math::
            log_cdf(x) = \log(cdf(x))
        """
        return self.log(self._call_cdf(value, *args, **kwargs))

    def survival_function(self, value, *args, **kwargs):
        """
        Evaluate the survival function at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its dist_spec_args through
            `args` or `kwargs`.
        """
        return self._call_survival(value, *args, **kwargs)

    def _calc_survival_from_call_cdf(self, value, *args, **kwargs):
        r"""
        Evaluate survival function from cdf.

        .. math::
            survival_function(x) =  1 - (cdf(x))
        """
        return 1.0 - self._call_cdf(value, *args, **kwargs)

    def _calc_survival_from_log_survival(self, value, *args, **kwargs):
        r"""
        Evaluate survival function from log survival function.

        .. math::
            survival(x) = \exp(survival_function(x))
        """
        return self.exp(self._log_survival(value, *args, **kwargs))

    def log_survival(self, value, *args, **kwargs):
        """
        Evaluate the log survival function at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its dist_spec_args through
            `args` or `kwargs`.
        """
        return self._call_log_survival(value, *args, **kwargs)

    def _calc_log_survival_from_call_survival(self, value, *args, **kwargs):
        r"""
        Evaluate log survival function from survival function.

        .. math::
            log_survival(x) = \log(survival_function(x))
        """
        return self.log(self._call_survival(value, *args, **kwargs))

    def kl_loss(self, dist, *args, **kwargs):
        """
        Evaluate the KL divergence, i.e. KL(a||b).

        Args:
            dist (str): type of the distribution.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            dist_spec_args of distribution b must be passed to the function through `args` or `kwargs`.
            Passing in dist_spec_args of distribution a is optional.
        """
        return self._kl_loss(dist, *args, **kwargs)

    def mean(self, *args, **kwargs):
        """
        Evaluate the mean.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            *args* or *kwargs*.
        """
        return self._mean(*args, **kwargs)

    def mode(self, *args, **kwargs):
        """
        Evaluate the mode.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            *args* or *kwargs*.
        """
        return self._mode(*args, **kwargs)

    def sd(self, *args, **kwargs):
        """
        Evaluate the standard deviation.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            *args* or *kwargs*.
        """
        return self._call_sd(*args, **kwargs)

    def var(self, *args, **kwargs):
        """
        Evaluate the variance.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            *args* or *kwargs*.
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

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            *args* or *kwargs*.
        """
        return self._entropy(*args, **kwargs)

    def cross_entropy(self, dist, *args, **kwargs):
        """
        Evaluate the cross_entropy between distribution a and b.

        Args:
            dist (str): type of the distribution.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            dist_spec_args of distribution b must be passed to the function through `args` or `kwargs`.
            Passing in dist_spec_args of distribution a is optional.
        """
        return self._call_cross_entropy(dist, *args, **kwargs)

    def _calc_cross_entropy(self, dist, *args, **kwargs):
        r"""
        Evaluate cross_entropy from entropy and kl divergence.

        .. math::
            H(X, Y) = H(X) + KL(X||Y)
        """
        return self._entropy(*args, **kwargs) + self._kl_loss(dist, *args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Sampling function.

        Args:
            shape (tuple): shape of the sample.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dictionary): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            *args* or *kwargs*.
        """
        return self._sample(*args, **kwargs)

    def construct(self, name, *args, **kwargs):
        """
        Override `construct` in Cell.

        Note:
            Names of supported functions include:
            'prob', 'log_prob', 'cdf', 'log_cdf', 'survival_function', 'log_survival'
            'var', 'sd', 'entropy', 'kl_loss', 'cross_entropy', and 'sample'.

        Args:
            name (str): The name of the function.
            *args (list): A list of positional arguments that the function needs.
            **kwargs (dictionary): A dictionary of keyword arguments that the function needs.
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
