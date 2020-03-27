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

"""Component that Check if the function raises the expected Exception."""

import sys
import pytest

from ...components.icomponent import IExectorComponent
from ...utils import keyword

class CheckExceptionsEC(IExectorComponent):
    """
    Check if the function raises the expected Exception.

    Examples:
        {
            'block': f,
            'exception': Exception
        }
    """
    def run_function(self, function, inputs, verification_set):
        f = function[keyword.block]
        args = inputs[keyword.desc_inputs]
        e = function.get(keyword.exception, Exception)
        try:
            with pytest.raises(e):
                f(*args)
        except:
            raise Exception(f"Expect {e}, but got {sys.exc_info()[0]}")
