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
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
import mindspore.nn as nn
from .distribution import Distribution
from ._utils.utils import check_type, raise_not_impl_error
from ._utils.custom_ops import exp_generic, log_generic

class TransformedDistribution(Distribution):
    """
    Transformed Distribution.
    This class contains a bijector and a distribution and transforms the original distribution
    to a new distribution through the operation defined by the bijector.

    Args:
        bijector (Bijector): transformation to perform.
        distribution (Distribution): The original distribution.
        name (str): name of the transformed distribution. Default: transformed_distribution.

    Note:
        The arguments used to initialize the original distribution cannot be None.
        For example, mynormal = nn.Normal(dtype=dtyple.float32) cannot be used to initialized a
        TransformedDistribution since mean and sd are not specified.

    Examples:
        >>> # To initialize a transformed distribution, e.g. lognormal distribution,
        >>> # using Normal distribution as the base distribution, and Exp bijector as the bijector function.
        >>> import mindspore.nn.probability.distribution as msd
        >>> import mindspore.nn.probability.bijector as msb
        >>> ln = msd.TransformedDistribution(msb.Exp(),
        >>>                                  msd.Normal(0.0, 1.0, dtype=mstype.float32),
        >>>                                  dtype=mstype.float32)
        >>>
        >>> # To use a transformed distribution in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.ln = msd.TransformedDistribution(msb.Exp(),
        >>>                                               msd.Normal(0.0, 1.0, dtype=mstype.float32),
        >>>                                               dtype=mstype.float32)
        >>>
        >>>     def construct(self, value):
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'sample' with the name of the function
        >>>         ans = self.ln.sample(shape=(2, 3))
    """
    def __init__(self,
                 bijector,
                 distribution,
                 dtype,
                 seed=0,
                 name="transformed_distribution"):
        """
        Constructor of transformed_distribution class.
        """
        param = dict(locals())
        validator.check_value_type('bijector', bijector, [nn.probability.bijector.Bijector], type(self).__name__)
        validator.check_value_type('distribution', distribution, [Distribution], type(self).__name__)
        valid_dtype = mstype.number_type
        check_type(dtype, valid_dtype, type(self).__name__)
        super(TransformedDistribution, self).__init__(seed, dtype, name, param)

        self._bijector = bijector
        self._distribution = distribution
        self._is_linear_transformation = bijector.is_constant_jacobian
        self.exp = exp_generic
        self.log = log_generic

    @property
    def bijector(self):
        return self._bijector

    @property
    def distribution(self):
        return self._distribution

    @property
    def is_linear_transformation(self):
        return self._is_linear_transformation

    def _cdf(self, *args, **kwargs):
        r"""
        .. math::
            Y = g(X)
            P(Y <= a) = P(X <= g^{-1}(a))
        """
        inverse_value = self.bijector("inverse", *args, **kwargs)
        return self.distribution("cdf", inverse_value)

    def _log_cdf(self, *args, **kwargs):
        return self.log(self._cdf(*args, **kwargs))

    def _survival_function(self, *args, **kwargs):
        return 1.0 - self._cdf(*args, **kwargs)

    def _log_survival(self, *args, **kwargs):
        return self.log(self._survival_function(*args, **kwargs))

    def _log_prob(self, *args, **kwargs):
        r"""
        .. math::
            Y = g(X)
            Py(a) = Px(g^{-1}(a)) * (g^{-1})'(a)
            \log(Py(a)) = \log(Px(g^{-1}(a))) + \log((g^{-1})'(a))
        """
        inverse_value = self.bijector("inverse", *args, **kwargs)
        unadjust_prob = self.distribution("log_prob", inverse_value)
        log_jacobian = self.bijector("inverse_log_jacobian", *args, **kwargs)
        return unadjust_prob + log_jacobian

    def _prob(self, *args, **kwargs):
        return self.exp(self._log_prob(*args, **kwargs))

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
