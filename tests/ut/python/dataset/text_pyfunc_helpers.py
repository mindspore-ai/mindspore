# Copyright 2026 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Private Python inputs for retained data-source and cache pipeline tests."""
import numpy as np


def _string(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def case_fold(value):
    """Fold the fixture's UTF-8 strings without a native tokenizer dependency."""
    array = np.asarray(value)
    return np.asarray([_string(x).casefold() for x in array.reshape(-1)]).reshape(array.shape)


def split_words(value):
    """Supply the existing cache test's split/empty-string input contract."""
    words = _string(np.asarray(value).item()).split()
    return np.asarray(words or [""], dtype=np.str_)
