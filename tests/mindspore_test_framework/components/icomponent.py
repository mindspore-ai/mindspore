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

"""Component interfaces."""


class IComponent:
    """Component interfaces."""

    def __init__(self, verification_set):
        self.verification_set = verification_set

    def __call__(self):
        raise NotImplementedError


class IDataComponent(IComponent):
    """Create inputs for verification_set."""

    def __call__(self):
        raise NotImplementedError


class IBuilderComponent(IComponent):
    """Build system under test."""

    def __call__(self):
        raise NotImplementedError


class IExectorComponent(IComponent):
    """Execute sut, take (function, input) pairs as input."""

    def __init__(self, verification_set, function, inputs):
        super(IExectorComponent, self).__init__(verification_set)
        self.function = function
        self.inputs = inputs

    def __call__(self):
        raise NotImplementedError


class IVerifierComponent(IComponent):
    """Verify sut result, take (expect, result) pairs as input."""

    def __init__(self, verification_set, expect, result):
        super(IVerifierComponent, self).__init__(verification_set)
        self.expect = expect
        self.func_result = result

    def __call__(self):
        raise NotImplementedError


class IFIPolicyComponent(IComponent):
    """Combine functions/inputs."""

    def __init__(self, verification_set, function, inputs):
        super(IFIPolicyComponent, self).__init__(verification_set)
        self.function = function
        self.inputs = inputs

    def __call__(self):
        raise NotImplementedError


class IERPolicyComponent(IComponent):
    """Combine expects and results."""

    def __init__(self, verification_set, expect, result):
        super(IERPolicyComponent, self).__init__(verification_set)
        self.expect = expect
        self.result = result

    def __call__(self):
        raise NotImplementedError


class IFacadeComponent(IComponent):
    """Adapt verification_set."""

    def __call__(self):
        raise NotImplementedError
