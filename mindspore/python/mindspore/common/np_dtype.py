# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Numpy data type for MindSpore."""

from mindspore._c_expression.np_dtypes import np_version_valid
if np_version_valid(True):
    from mindspore._c_expression.np_dtypes import bfloat16 # pylint: disable=unused-import

__all__ = []
if np_version_valid(False):
    __all__.extend(["bfloat16"])
