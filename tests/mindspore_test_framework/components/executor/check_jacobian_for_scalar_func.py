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

"""Component that check jacobian against numeric with respect to inputs for scalar_func."""

from ...components.icomponent import IExectorComponent
from ...utils.check_gradient import check_jacobian, ScalarGradChecker
from ...utils.config_util import get_grad_checking_options


class CheckJacobianForScalarFunctionEC(IExectorComponent):
    """
    Check jacobian against numeric with respect to inputs for scalar_func, execute and verify.

    Examples:
        'block': scalar_function
    """

    def __call__(self):
        f, args, delta, max_error, input_selector, output_selector, _, _ = \
            get_grad_checking_options(self.function, self.inputs)
        check_jacobian(f, *args, delta=delta, max_error=max_error, grad_checker_class=ScalarGradChecker,
                       input_selector=input_selector, output_selector=output_selector)
