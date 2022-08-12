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
# See the License foNtest_resr the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Built-in validators."""
from __future__ import absolute_import

from functools import wraps

from mindspore.dataset.core.validator_helpers import parse_user_args, check_pos_int32


def check_callback(method):
    """check the input arguments of DSCallback."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [step_size], _ = parse_user_args(method, *args, **kwargs)
        check_pos_int32(step_size, "step_size")
        return method(self, *args, **kwargs)

    return new_method
