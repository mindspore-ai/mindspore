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
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.ops.primitive import constexpr
from mindspore.ops.operations import _inner_ops as inner
from mindspore._checkparam import Validator as validator
from ._utils.utils import raise_none_error, cast_to_tensor, set_param_type, cast_type_for_device,\
    raise_not_implemented_util
from ._utils.utils import CheckTuple, CheckTensor
from ._utils.custom_ops import broadcast_to, exp_generic, log_generic


class Distribution(Cell):
    """
    Base class for all mathematical distributions.

    Args:
        seed (int): The seed is used in sampling. 0 is used if it is None.
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

    Supported Platforms:
        ``Ascend`` ``GPU``
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
            seed = 0
        validator.check_value_type('name', name, [str], type(self).__name__)
        validator.check_non_negative_int(seed, 'seed', name)

        self._name = name
        self._seed = seed
        self._dtype = cast_type_for_device(dtype)
        self._parameters = {}
        self.default_parameters = []
        self.parameter_names = []

        # parsing parameters
        for k in param.keys():
            if not(k == 'self' or k.startswith('_')):
                self._parameters[k] = param[k]

        # if not a transformed distribution, set the following attribute
        if 'distribution' not in self.parameters.keys():
            self.parameter_type = set_param_type(
                self.parameters.get('param_dict', {}), dtype)
            self._batch_shape = self._calc_batch_shape()
            self._is_scalar_batch = self._check_is_scalar_batch()
            self._broadcast_shape = self._batch_shape

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
        self.device_target = context.get_context('device_target')
        self.checktuple = CheckTuple()

        @constexpr(check=False)
        def _check_tensor(x, name):
            CheckTensor()(x, name)
            return x
        # we use constexpr to force CheckTensor to run only once in pynative mode
        self.checktensor = CheckTensor() if self.context_mode == 0 else _check_tensor
        self.broadcast = broadcast_to

        # ops needed for the base class
        self.cast_base = P.Cast()
        self.dtype_base = P.DType()
        self.fill_base = P.Fill()
        self.sametypeshape_base = inner.SameTypeShape()
        self.sq_base = P.Square()
        self.sqrt_base = P.Sqrt()
        self.shape_base = P.Shape()
        if self.device_target != "Ascend":
            self.log_base = P.Log()
            self.exp_base = P.Exp()
        else:
            self.exp_base = exp_generic
            self.log_base = log_generic

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
    def batch_shape(self):
        return self._batch_shape

    @property
    def broadcast_shape(self):
        return self._broadcast_shape

    def _reset_parameters(self):
        self.default_parameters = []
        self.parameter_names = []

    def _add_parameter(self, value, name):
        """
        Cast `value` to a tensor and add it to `self.default_parameters`.
        Add `name` into  and `self.parameter_names`.
        """
        # initialize the attributes if they do not exist yet
        if not hasattr(self, 'default_parameters'):
            self.default_parameters = []
            self.parameter_names = []
        # cast value to a tensor if it is not None
        value_t = None if value is None else cast_to_tensor(value, self.parameter_type)
        self.default_parameters.append(value_t)
        self.parameter_names.append(name)
        return value_t

    def _check_param_type(self, *args):
        """
        Check the availability and validity of default parameters and `dist_spec_args`.
        `dist_spec_args` passed in must be tensors. If default parameters of the distribution
        are None, the parameters must be passed in through `args`.
        """
        broadcast_shape = None
        broadcast_shape_tensor = None
        common_dtype = None
        out = []

        for arg, name, default in zip(args, self.parameter_names, self.default_parameters):
            # check if the argument is a Tensor
            if arg is not None:
                self.checktensor(arg, name)
            else:
                arg = default if default is not None else raise_none_error(name)

            # broadcast if the number of args > 1
            if broadcast_shape is None:
                broadcast_shape = self.shape_base(arg)
                common_dtype = self.dtype_base(arg)
                broadcast_shape_tensor = self.fill_base(
                    common_dtype, broadcast_shape, 1.0)
            else:
                broadcast_shape = self.shape_base(arg + broadcast_shape_tensor)
                broadcast_shape_tensor = self.fill_base(
                    common_dtype, broadcast_shape, 1.0)
                arg = self.broadcast(arg, broadcast_shape_tensor)
                # check if the arguments have the same dtype
                self.sametypeshape_base(arg, broadcast_shape_tensor)

            arg = self.cast_base(arg, self.parameter_type)
            out.append(arg)

        if len(out) == 1:
            return out[0]

        # broadcast all args to broadcast_shape
        result = ()
        for arg in out:
            arg = self.broadcast(arg, broadcast_shape_tensor)
            result = result + (arg,)
        return result

    def _check_value(self, value, name):
        """
        Check availability of `value` as a Tensor.
        """
        self.checktensor(value, name)
        return value

    def _check_is_scalar_batch(self):
        """
        Check if the parameters used during initialization are scalars.
        """
        param_dict = self.parameters.get('param_dict')
        for value in param_dict.values():
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                return False
        return True

    def _calc_batch_shape(self):
        """
        Calculate the broadcast shape of the parameters used during initialization.
        """
        broadcast_shape_tensor = None
        param_dict = self.parameters.get('param_dict')
        for value in param_dict.values():
            if value is None:
                return None
            if broadcast_shape_tensor is None:
                broadcast_shape_tensor = cast_to_tensor(value)
            else:
                value = cast_to_tensor(value)
                broadcast_shape_tensor = (value + broadcast_shape_tensor)
        return broadcast_shape_tensor.shape

    def _set_prob(self):
        """
        Set probability function based on the availability of `_prob` and `_log_likehood`.
        """
        if hasattr(self, '_prob'):
            self._call_prob = self._prob
        elif hasattr(self, '_log_prob'):
            self._call_prob = self._calc_prob_from_log_prob
        else:
            self._call_prob = self._raise_not_implemented_error('prob')

    def _set_sd(self):
        """
        Set standard deviation based on the availability of `_sd` and `_var`.
        """
        if hasattr(self, '_sd'):
            self._call_sd = self._sd
        elif hasattr(self, '_var'):
            self._call_sd = self._calc_sd_from_var
        else:
            self._call_sd = self._raise_not_implemented_error('sd')

    def _set_var(self):
        """
        Set variance based on the availability of `_sd` and `_var`.
        """
        if hasattr(self, '_var'):
            self._call_var = self._var
        elif hasattr(self, '_sd'):
            self._call_var = self._calc_var_from_sd
        else:
            self._call_var = self._raise_not_implemented_error('var')

    def _set_log_prob(self):
        """
        Set log probability based on the availability of `_prob` and `_log_prob`.
        """
        if hasattr(self, '_log_prob'):
            self._call_log_prob = self._log_prob
        elif hasattr(self, '_prob'):
            self._call_log_prob = self._calc_log_prob_from_prob
        else:
            self._call_log_prob = self._raise_not_implemented_error('log_prob')

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
        else:
            self._call_cdf = self._raise_not_implemented_error('cdf')

    def _set_survival(self):
        """
        Set survival function based on the availability of _survival function and `_log_survival`
        and `_call_cdf`.
        """
        if not (hasattr(self, '_survival_function') or hasattr(self, '_log_survival') or
                hasattr(self, '_cdf') or hasattr(self, '_log_cdf')):
            self._call_survival = self._raise_not_implemented_error(
                'survival_function')
        elif hasattr(self, '_survival_function'):
            self._call_survival = self._survival_function
        elif hasattr(self, '_log_survival'):
            self._call_survival = self._calc_survival_from_log_survival
        elif hasattr(self, '_call_cdf'):
            self._call_survival = self._calc_survival_from_call_cdf

    def _set_log_cdf(self):
        """
        Set log cdf based on the availability of `_log_cdf` and `_call_cdf`.
        """
        if not (hasattr(self, '_log_cdf') or hasattr(self, '_cdf') or
                hasattr(self, '_survival_function') or hasattr(self, '_log_survival')):
            self._call_log_cdf = self._raise_not_implemented_error('log_cdf')
        elif hasattr(self, '_log_cdf'):
            self._call_log_cdf = self._log_cdf
        elif hasattr(self, '_call_cdf'):
            self._call_log_cdf = self._calc_log_cdf_from_call_cdf

    def _set_log_survival(self):
        """
        Set log survival based on the availability of `_log_survival` and `_call_survival`.
        """
        if not (hasattr(self, '_log_survival') or hasattr(self, '_survival_function') or
                hasattr(self, '_log_cdf') or hasattr(self, '_cdf')):
            self._call_log_survival = self._raise_not_implemented_error(
                'log_cdf')
        elif hasattr(self, '_log_survival'):
            self._call_log_survival = self._log_survival
        elif hasattr(self, '_call_survival'):
            self._call_log_survival = self._calc_log_survival_from_call_survival

    def _set_cross_entropy(self):
        """
        Set log survival based on the availability of `_cross_entropy`.
        """
        if hasattr(self, '_cross_entropy'):
            self._call_cross_entropy = self._cross_entropy
        else:
            self._call_cross_entropy = self._raise_not_implemented_error(
                'cross_entropy')

    def _get_dist_args(self, *args, **kwargs):
        return raise_not_implemented_util('get_dist_args', self.name, *args, **kwargs)

    def get_dist_args(self, *args, **kwargs):
        """
        Check the availability and validity of default parameters and `dist_spec_args`.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            `dist_spec_args` must be passed in through list or dictionary. The order of `dist_spec_args`
            should follow the initialization order of default parameters through `_add_parameter`.
            If some `dist_spec_args` is None, the corresponding default parameter is returned.

        Return:
            list[Tensor], the list of parameters.
        """
        return self._get_dist_args(*args, **kwargs)

    def _get_dist_type(self):
        return raise_not_implemented_util('get_dist_type', self.name)

    def get_dist_type(self):
        """
        Return the type of the distribution.

        Return:
            string, the name of distribution.
        """
        return self._get_dist_type()

    def _raise_not_implemented_error(self, func_name):
        name = self.name

        def raise_error(*args, **kwargs):
            return raise_not_implemented_util(func_name, name, *args, **kwargs)
        return raise_error

    def log_prob(self, value, *args, **kwargs):
        """
        Evaluate the log probability(pdf or pmf) at the given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its `dist_spec_args` through
            `args` or `kwargs`.

        Return:
            Tensor, the value of log probability.
        """
        return self._call_log_prob(value, *args, **kwargs)

    def _calc_prob_from_log_prob(self, value, *args, **kwargs):
        r"""
        Evaluate prob from log probability.

        .. math::
            probability(x) = \exp(log_likehood(x))
        """
        return self.exp_base(self._log_prob(value, *args, **kwargs))

    def prob(self, value, *args, **kwargs):
        """
        Evaluate the probability (pdf or pmf) at given value. For a discrete distribution,
        it is a probability mass function, while for a continuous distribution, it is probability density function.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its `dist_spec_args` through
            `args` or `kwargs`.

        Return:
            Tensor, the value of probability.
        """
        return self._call_prob(value, *args, **kwargs)

    def _calc_log_prob_from_prob(self, value, *args, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            log_prob(x) = \log(prob(x))
        """
        return self.log_base(self._prob(value, *args, **kwargs))

    def cdf(self, value, *args, **kwargs):
        """
        Evaluate the cumulative distribution function(cdf) at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its `dist_spec_args` through
            `args` or `kwargs`.

        Return:
            Tensor, the cdf of the distribution.
        """
        return self._call_cdf(value, *args, **kwargs)

    def _calc_cdf_from_log_cdf(self, value, *args, **kwargs):
        r"""
        Evaluate cdf from log_cdf.

        .. math::
            cdf(x) = \exp(log_cdf(x))
        """
        return self.exp_base(self._log_cdf(value, *args, **kwargs))

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
        return 1.0 - self.exp_base(self._log_survival(value, *args, **kwargs))

    def log_cdf(self, value, *args, **kwargs):
        """
        Evaluate the log the cumulative distribution function(cdf) at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its `dist_spec_args` through
            `args` or `kwargs`.

        Return:
            Tensor, the log cdf of the distribution.
        """
        return self._call_log_cdf(value, *args, **kwargs)

    def _calc_log_cdf_from_call_cdf(self, value, *args, **kwargs):
        r"""
        Evaluate log cdf from cdf.

        .. math::
            log_cdf(x) = \log(cdf(x))
        """
        return self.log_base(self._call_cdf(value, *args, **kwargs))

    def survival_function(self, value, *args, **kwargs):
        """
        Evaluate the survival function at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its `dist_spec_args` through
            `args` or `kwargs`.

        Return:
            Tensor, the survival function of the distribution.
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
        return self.exp_base(self._log_survival(value, *args, **kwargs))

    def log_survival(self, value, *args, **kwargs):
        """
        Evaluate the log survival function at given value.

        Args:
            value (Tensor): value to be evaluated.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its `dist_spec_args` through
            `args` or `kwargs`.

        Return:
            Tensor, the log survival function of the distribution.
        """
        return self._call_log_survival(value, *args, **kwargs)

    def _calc_log_survival_from_call_survival(self, value, *args, **kwargs):
        r"""
        Evaluate log survival function from survival function.

        .. math::
            log_survival(x) = \log(survival_function(x))
        """
        return self.log_base(self._call_survival(value, *args, **kwargs))

    def _kl_loss(self, *args, **kwargs):
        return raise_not_implemented_util('kl_loss', self.name, *args, **kwargs)

    def kl_loss(self, dist, *args, **kwargs):
        """
        Evaluate the KL divergence, i.e. KL(a||b).

        Args:
            dist (str): type of the distribution.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            `dist_spec_args` of distribution b must be passed to the function through `args` or `kwargs`.
            Passing in `dist_spec_args` of distribution a is optional.

        Return:
            Tensor, the kl loss function of the distribution.
        """
        return self._kl_loss(dist, *args, **kwargs)

    def _mean(self, *args, **kwargs):
        return raise_not_implemented_util('mean', self.name, *args, **kwargs)

    def mean(self, *args, **kwargs):
        """
        Evaluate the mean.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            `args` or `kwargs`.

        Return:
            Tensor, the mean of the distribution.
        """
        return self._mean(*args, **kwargs)

    def _mode(self, *args, **kwargs):
        return raise_not_implemented_util('mode', self.name, *args, **kwargs)

    def mode(self, *args, **kwargs):
        """
        Evaluate the mode.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            `args` or `kwargs`.

        Return:
            Tensor, the mode of the distribution.
        """
        return self._mode(*args, **kwargs)

    def sd(self, *args, **kwargs):
        """
        Evaluate the standard deviation.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            `args` or `kwargs`.

        Return:
            Tensor, the standard deviation of the distribution.
        """
        return self._call_sd(*args, **kwargs)

    def var(self, *args, **kwargs):
        """
        Evaluate the variance.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            `args` or `kwargs`.

        Return:
            Tensor, the variance of the distribution.
        """
        return self._call_var(*args, **kwargs)

    def _calc_sd_from_var(self, *args, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            STD(x) = \sqrt(VAR(x))
        """
        return self.sqrt_base(self._var(*args, **kwargs))

    def _calc_var_from_sd(self, *args, **kwargs):
        r"""
        Evaluate log probability from probability.

        .. math::
            VAR(x) = STD(x) ^ 2
        """
        return self.sq_base(self._sd(*args, **kwargs))

    def _entropy(self, *args, **kwargs):
        return raise_not_implemented_util('entropy', self.name, *args, **kwargs)

    def entropy(self, *args, **kwargs):
        """
        Evaluate the entropy.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            `args` or `kwargs`.

        Return:
            Tensor, the entropy of the distribution.
        """
        return self._entropy(*args, **kwargs)

    def cross_entropy(self, dist, *args, **kwargs):
        """
        Evaluate the cross_entropy between distribution a and b.

        Args:
            dist (str): type of the distribution.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            `dist_spec_args` of distribution b must be passed to the function through `args` or `kwargs`.
            Passing in `dist_spec_args` of distribution a is optional.

        Return:
            Tensor, the cross_entropy of two distributions.
        """
        return self._call_cross_entropy(dist, *args, **kwargs)

    def _calc_cross_entropy(self, dist, *args, **kwargs):
        r"""
        Evaluate cross_entropy from entropy and kl divergence.

        .. math::
            H(X, Y) = H(X) + KL(X||Y)
        """
        return self._entropy(*args, **kwargs) + self._kl_loss(dist, *args, **kwargs)

    def _sample(self, *args, **kwargs):
        return raise_not_implemented_util('sample', self.name, *args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Sampling function.

        Args:
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Note:
            A distribution can be optionally passed to the function by passing its *dist_spec_args* through
            `args` or `kwargs`.

        Return:
            Tensor, the sample generated from the distribution.
        """
        return self._sample(*args, **kwargs)

    def construct(self, name, *args, **kwargs):
        """
        Override `construct` in Cell.

        Note:
            Names of supported functions include:
            'prob', 'log_prob', 'cdf', 'log_cdf', 'survival_function', 'log_survival',
            'var', 'sd', 'mode', 'mean', 'entropy', 'kl_loss', 'cross_entropy', 'sample',
            'get_dist_args', and 'get_dist_type'.

        Args:
            name (str): The name of the function.
            *args (list): A list of positional arguments that the function needs.
            **kwargs (dict): A dictionary of keyword arguments that the function needs.

        Return:
            Tensor, the value of corresponding computation method.
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
        if name == 'get_dist_args':
            return self._get_dist_args(*args, **kwargs)
        if name == 'get_dist_type':
            return self._get_dist_type()
        return raise_not_implemented_util(name, self.name, *args, **kwargs)
