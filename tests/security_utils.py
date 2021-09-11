# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Security utils."""
from functools import wraps

from mindspore._c_expression import security


def security_off_wrap(func):
    """Wrapper for tests which do not need to run security on."""

    @wraps(func)
    def pass_test_when_security_on(*args, **kwargs):
        if security.enable_security():
            return None
        return func(*args, **kwargs)

    return pass_test_when_security_on
