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
"""Utility functions to help bnn layers."""
from mindspore.common.tensor import Tensor
from ...cell import Cell


def check_prior(prior_fn, arg_name):
    """check prior distribution of bnn layers."""
    if isinstance(prior_fn, Cell):
        prior = prior_fn
    else:
        prior = prior_fn()
    for prior_name, prior_dist in prior.name_cells().items():
        if prior_name != 'normal':
            raise TypeError(f"The type of distribution of `{arg_name}` must be `normal`")
        if not (isinstance(getattr(prior_dist, '_mean_value'), Tensor) and
                isinstance(getattr(prior_dist, '_sd_value'), Tensor)):
            raise TypeError(f"The input form of `{arg_name}` is incorrect")
    return prior


def check_posterior(posterior_fn, shape, param_name, arg_name):
    """check posterior distribution of bnn layers."""
    try:
        posterior = posterior_fn(shape=shape, name=param_name)
    except TypeError:
        raise TypeError(f'The type of `{arg_name}` must be `NormalPosterior`')
    finally:
        pass
    for posterior_name, _ in posterior.name_cells().items():
        if posterior_name != 'normal':
            raise TypeError(f"The type of distribution of `{arg_name}` must be `normal`")
    return posterior
