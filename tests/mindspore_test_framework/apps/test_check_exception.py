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

"""Checking exception."""

from ..mindspore_test import mindspore_test
from ..pipeline.forward.verify_exception import pipeline_for_verify_exception_for_case_by_case_config


def func_raise_exception(x, y):
    raise ValueError()


verification_set = [
    ('func_raise_exception', {
        'block': (func_raise_exception, {'exception': ValueError}),
        'desc_inputs': [[1, 1], [2, 2]],
    })
]


@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    return verification_set
