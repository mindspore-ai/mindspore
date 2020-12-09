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

"""Base class `PerturbationAttribtuion`"""

from mindspore.train._utils import check_value_type
from mindspore.nn import Cell

from ..attribution import Attribution


class PerturbationAttribution(Attribution):
    """
    Base class for perturbation-based attribution methods.

    All perturbation-based _attribution methods extend from this class.
    """

    def __init__(self,
                 network,
                 activation_fn,
                 perturbation_per_eval,
                 ):
        super(PerturbationAttribution, self).__init__(network)
        check_value_type("activation_fn", activation_fn, Cell)
        self._activation_fn = activation_fn
        check_value_type('perturbation_per_eval', perturbation_per_eval, int)
        if perturbation_per_eval <= 0:
            raise ValueError('Argument perturbation_per_eval should be a positive integer.')
        self._perturbation_per_eval = perturbation_per_eval
