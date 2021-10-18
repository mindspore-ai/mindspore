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

"""Verification pipeline engine."""

import logging
import pytest

from .components.icomponent import IDataComponent, IBuilderComponent, IExectorComponent, \
    IVerifierComponent, IFIPolicyComponent, IERPolicyComponent, IComponent, \
    IFacadeComponent
from .utils import keyword


def mindspore_test(verification_pipeline):
    """
    Run verification pipeline.

    Args:
        verification_pipeline (list): Pipeline designed to do verification.

    Returns:
    """

    def decorate(get_verification_set):
        verification_set = get_verification_set()

        facade_components = []
        data_components = []
        builder_components = []
        executor_components = []
        verifier_components = []
        fi_policy_components = []
        er_policy_components = []
        for component in verification_pipeline:
            if issubclass(component, IFacadeComponent):
                facade_components.append(component)
            elif issubclass(component, IDataComponent):
                data_components.append(component)
            elif issubclass(component, IBuilderComponent):
                builder_components.append(component)
            elif issubclass(component, IExectorComponent):
                executor_components.append(component)
            elif issubclass(component, IVerifierComponent):
                verifier_components.append(component)
            elif issubclass(component, IFIPolicyComponent):
                fi_policy_components.append(component)
            elif issubclass(component, IERPolicyComponent):
                er_policy_components.append(component)
            else:
                raise Exception(f'{component} is not an instance of {IComponent}')

        for component in facade_components:
            fc = component(verification_set)
            verification_set = fc()

        inputs = []
        for component in data_components:
            dc = component(verification_set)
            item = dc()
            inputs.extend(item)

        if not inputs:
            logging.warning("Inputs set is empty.")

        functions = []
        for component in builder_components:
            bc = component(verification_set)
            f = bc()
            functions.extend(f)

        if not functions:
            logging.warning("Function set is empty.")

        fis = []
        for component in fi_policy_components:
            fipc = component(verification_set, functions, inputs)
            result = fipc()
            fis.extend(result)

        if not fis:
            logging.warning("Function inputs pair set is empty.")

        def test_case(args):
            sut, inputs = args

            results = []
            for component in executor_components:
                ec = component(verification_set, sut, inputs)
                result = ec()
                results.append(result)

            if not results:
                logging.warning("Result set is empty.")

            expect_actuals = []
            for component in er_policy_components:
                erpc = component(verification_set, verification_set['expect'], results)
                result = erpc()
                expect_actuals.extend(result)

            if not expect_actuals:
                logging.warning("Expect Result pair set is empty.")

            for ea in expect_actuals:
                for component in verifier_components:
                    vc = component(verification_set, *ea)
                    vc()

        def get_tc_name(f, inputs):
            tc_id = f[keyword.id] + '-' + inputs[keyword.id]
            group = f[keyword.group] + '-' + inputs[keyword.group]
            return 'Group_' + group + '-' + 'Id_' + tc_id

        if fis:
            m = pytest.mark.parametrize('args', fis, ids=lambda fi: get_tc_name(*fi))(test_case)
            m.__orig__ = get_verification_set
            return m

    return decorate
