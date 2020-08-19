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

from ..core.validator_helpers import type_check, check_uint32, check_uint64

class DatasetCache:
    """
    A client to interface with tensor caching service
    """

    def __init__(self, session_id=None, size=0, spilling=False, port=50052, prefetch_size=20):
        check_uint32(session_id, "session_id")
        check_uint64(size, "size")
        type_check(spilling, (bool,), "spilling")
        check_uint32(port, "port")
        check_uint32(prefetch_size, "prefetch size")

        self.session_id = session_id
        self.size = size
        self.spilling = spilling
        self.port = port
        self.prefetch_size = prefetch_size
        self.cache_client = CacheClient(session_id, size, spilling, port, prefetch_size)

    def GetStat(self):
        return self.cache_client.GetStat()

    def __deepcopy__(self, memodict):
        if id(self) in memodict:
            return memodict[id(self)]
        cls = self.__class__
        new_cache = cls.__new__(cls)
        memodict[id(self)] = new_cache
        new_cache.session_id = copy.deepcopy(self.session_id, memodict)
        new_cache.spilling = copy.deepcopy(self.spilling, memodict)
        new_cache.size = copy.deepcopy(self.size, memodict)
        new_cache.port = copy.deepcopy(self.port, memodict)
        new_cache.prefetch_size = copy.deepcopy(self.prefetch_size, memodict)
        new_cache.cache_client = self.cache_client
        return new_cache
