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

    def run(self):
        raise NotImplementedError

    def get_result(self):
        return self.result


class IDataComponent(IComponent):
    """Create inputs for verification_set."""
    def run(self):
        self.result = self.create_inputs(self.verification_set)

    def create_inputs(self, verification_set):
        raise NotImplementedError


class IBuilderComponent(IComponent):
    """Build system under test."""
    def run(self):
        self.result = self.build_sut(self.verification_set)

    def build_sut(self, verification_set):
        raise NotImplementedError


class IExectorComponent(IComponent):
    """Execute sut, take (function, input) pairs as input."""
    def __init__(self, verification_set, function, inputs):
        super(IExectorComponent, self).__init__(verification_set)
        self.function = function
        self.inputs = inputs

    def run(self):
        self.result = self.run_function(self.function, self.inputs, self.verification_set)

    def run_function(self, function, inputs, verification_set):
        raise NotImplementedError


class IVerifierComponent(IComponent):
    """Verify sut result, take (expect, result) pairs as input."""
    def __init__(self, verification_set, expect, result):
        super(IVerifierComponent, self).__init__(verification_set)
        self.expect = expect
        self.func_result = result

    def run(self):
        self.result = self.verify(self.expect, self.func_result, self.verification_set)

    def verify(self, expect, func_result, verification_set):
        raise NotImplementedError


class IFIPolicyComponent(IComponent):
    """Combine functions/inputs."""
    def __init__(self, verification_set, function, inputs):
        super(IFIPolicyComponent, self).__init__(verification_set)
        self.function = function
        self.inputs = inputs

    def run(self):
        self.result = self.combine(self.function, self.inputs, self.verification_set)

    def combine(self, function, inputs, verification_set):
        raise NotImplementedError


class IERPolicyComponent(IComponent):
    """Combine expects and results."""
    def __init__(self, verification_set, expect, result):
        super(IERPolicyComponent, self).__init__(verification_set)
        self.expect = expect
        self.result = result

    def run(self):
        self.result = self.combine(self.expect, self.result, self.verification_set)

    def combine(self, expect, result, verification_set):
        raise NotImplementedError


class IFacadeComponent(IComponent):
    """Adapt verification_set."""
    def run(self):
        self.result = self.adapt(self.verification_set)

    def adapt(self, verification_set):
        raise NotImplementedError
