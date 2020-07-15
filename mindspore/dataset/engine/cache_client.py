# Copyright 2019 Huawei Technologies Co., Ltd
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
"""Cache client
"""

import copy
from mindspore._c_dataengine import CacheClient

class DatasetCache:
    """
    A client to interface with tensor caching service
    """

    def __init__(self, session_id=None, size=None, spilling=False):
        if session_id is None:
            raise RuntimeError("Session generation is not implemented yet. session id required")
        self.size = size if size is not None else 0
        if size < 0:
            raise ValueError("cache size should be 0 or positive integer value but got: size={}".format(size))
        if not isinstance(spilling, bool):
            raise ValueError(
                "spilling argument for cache should be a boolean value but got: spilling={}".format(spilling))
        self.session_id = session_id
        self.spilling = spilling
        self.cache_client = CacheClient(session_id, size, spilling)

    def __deepcopy__(self, memodict):
        if id(self) in memodict:
            return memodict[id(self)]
        cls = self.__class__
        new_cache = cls.__new__(cls)
        memodict[id(self)] = new_cache
        new_cache.session_id = copy.deepcopy(self.session_id, memodict)
        new_cache.spilling = copy.deepcopy(self.spilling, memodict)
        new_cache.size = copy.deepcopy(self.size, memodict)
        new_cache.cache_client = self.cache_client
        return new_cache
