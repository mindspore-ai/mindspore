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
"""Transformed Distribution"""
import numpy as np
from mindspore._checkparam import Validator as validator
from mindspore.ops import operations as P
import mindspore.nn as nn
from .distribution import Distribution
from ._utils.utils import raise_not_impl_error
from ._utils.custom_ops import exp_generic, log_generic


class TransformedDistribution(Distribution):
    """
    Transformed Distribution.
    This class contains a bijector and a distribution and transforms the original distribution
    to a new distribution through the operation defined by the bijector.

    Args:
        bijector (Bijector): The transformation to perform.
        distribution (Distribution): The original distribution.
        seed (int): The seed is used in sampling. The global seed is used if it is None.
          If this seed is given when a TransformedDistribution object is initialised, the object's sampling function
          will use this seed; elsewise, the underlying distribution's seed will be used.
        name (str): The name of the transformed distribution. Default: 'transformed_distribution'.

    Note:
        The arguments used to initialize the original distribution cannot be None.
        For example, mynormal = nn.Normal(dtype=dtyple.float32) cannot be used to initialized a
        TransformedDistribution since `mean` and `sd` are not specified.

    Examples:
        >>> # To initialize a transformed distribution, e.g. a lognormal distribution,
        >>> # using a Normal distribution as the base distribution, and an Exp bijector as the bijector function.
        >>> import mindspore.nn.probability.distribution as msd
        >>> import mindspore.nn.probability.bijector as msb
        >>> ln = msd.TransformedDistribution(msb.Exp(),
        >>>                                  msd.Normal(0.0, 1.0, dtype=mstype.float32))
        >>>
        >>> # To use a transformed distribution in a network.
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.ln = msd.TransformedDistribution(msb.Exp(),
        >>>                                               msd.Normal(0.0, 1.0, dtype=mstype.float32))
        >>>
        >>>     def construct(self, value):
        >>>         # Similar calls can be made to other functions
        >>>         # by replacing 'sample' by the name of the function.
        >>>         ans = self.ln.sample(shape=(2, 3))
    """

    def __init__(self,
                 bijector,
                 distribution,
                 seed=None,
                 name="transformed_distribution"):
        """
        Constructor of transformed_distribution class.
        """
        param = dict(locals())
        validator.check_value_type('bijector', bijector,
                                   [nn.probability.bijector.Bijector], type(self).__name__)
        validator.check_value_type('distribution', distribution,
                                   [Distribution], type(self).__name__)
        super(TransformedDistribution, self).__init__(seed, distribution.dtype, name, param)

        self._bijector = bijector
        self._distribution = distribution
        self._is_linear_transformation = bijector.is_constant_jacobian
        self.default_parameters = distribution.default_parameters
        self.parameter_names = distribution.parameter_names
        self.exp = exp_generic
        self.log = log_generic
        self.equal_base = P.Equal()
        self.select_base = P.Select()

    @property
    def bijector(self):
        return self._bijector

    @property
    def distribution(self):
        return self._distribution

    @property
    def dtype(self):
        return self.distribution.dtype

    @property
    def is_linear_transformation(self):
        return self._is_linear_transformation

    def _cdf(self, value, *args, **kwargs):
        r"""
        .. math::
            Y = g(X)
            P(Y <= a) = P(X <= g^{-1}(a))
        """
        inverse_value = self.bijector("inverse", value)
        return self.distribution("cdf", inverse_value, *args, **kwargs)

    def _log_cdf(self, value, *args, **kwargs):
        return self.log(self._cdf(value, *args, **kwargs))

    def _survival_function(self, value, *args, **kwargs):
        return 1.0 - self._cdf(value, *args, **kwargs)

    def _log_survival(self, value, *args, **kwargs):
        return self.log(self._survival_function(value, *args, **kwargs))

    def _log_prob(self, value, *args, **kwargs):
        r"""
        .. math::
            Y = g(X)
            Py(a) = Px(g^{-1}(a)) * (g^{-1})'(a)
            \log(Py(a)) = \log(Px(g^{-1}(a))) + \log((g^{-1})'(a))
        """
        inverse_value = self.bijector("inverse", value)
        unadjust_prob = self.distribution("log_prob", inverse_value, *args, **kwargs)
        log_jacobian = self.bijector("inverse_log_jacobian", value)
        isneginf = self.equal_base(unadjust_prob, -np.inf)
        return self.select_base(isneginf, unadjust_prob, unadjust_prob + log_jacobian)

    def _prob(self, value, *args, **kwargs):
        return self.exp(self._log_prob(value, *args, **kwargs))

    def _sample(self, *args, **kwargs):
        org_sample = self.distribution("sample", *args, **kwargs)
        return self.bijector("forward", org_sample)

    def _mean(self, *args, **kwargs):
        """
        Note:
            This function maybe overridden by derived class.
        """
        if not self.is_linear_transformation:
            raise_not_impl_error("mean")

        return self.bijector("forward", self.distribution("mean", *args, **kwargs))
