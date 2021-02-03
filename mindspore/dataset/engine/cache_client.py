# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

from ..core.validator_helpers import type_check, check_uint32, check_uint64, check_positive, check_value


class DatasetCache:
    """
    A client to interface with tensor caching service.

    For details, please check `Chinese tutorial <https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_cache.html>`_,
    `Chinese programming guide <https://www.mindspore.cn/doc/programming_guide/zh-CN/master/cache.html?highlight=datasetcache>`_.

    Args:
        session_id (int): A user assigned session id for the current pipeline.
        size (int, optional): Size of the memory set aside for the row caching (default=0 which means unlimited,
            note that it might bring in the risk of running out of memory on the machine).
        spilling (bool, optional): Whether or not spilling to disk if out of memory (default=False).
        hostname (str, optional): Host name (default="127.0.0.1").
        port (int, optional): Port to connect to server (default=50052).
        num_connections (int, optional): Number of tcp/ip connections (default=12).
        prefetch_size (int, optional): Prefetch size (default=20).

    Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> # create a cache instance, in which session_id is generated from command line `cache_admin -g`
            >>> some_cache = ds.DatasetCache(session_id=session_id, size=0)
            >>>
            >>> dataset_dir = "path/to/imagefolder_directory"
            >>> ds1 = ds.ImageFolderDataset(dataset_dir, cache=some_cache)
    """

    def __init__(self, session_id, size=0, spilling=False, hostname=None, port=None, num_connections=None,
                 prefetch_size=None):
        check_uint32(session_id, "session_id")
        type_check(size, (int,), "size")
        if size != 0:
            check_positive(size, "size")
            check_uint64(size, "size")
        type_check(spilling, (bool,), "spilling")
        if hostname is not None:
            type_check(hostname, (str,), "hostname")
        if port is not None:
            type_check(port, (int,), "port")
            check_value(port, (1025, 65535), "port")
        if num_connections is not None:
            check_uint32(num_connections, "num_connections")
        if prefetch_size is not None:
            check_uint32(prefetch_size, "prefetch_size")

        self.session_id = session_id
        self.size = size
        self.spilling = spilling
        self.hostname = hostname
        self.port = port
        self.prefetch_size = prefetch_size
        self.num_connections = num_connections
        self.cache_client = CacheClient(session_id, size, spilling, hostname, port, num_connections, prefetch_size)

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
        new_cache.hostname = copy.deepcopy(self.hostname, memodict)
        new_cache.port = copy.deepcopy(self.port, memodict)
        new_cache.prefetch_size = copy.deepcopy(self.prefetch_size, memodict)
        new_cache.num_connections = copy.deepcopy(self.num_connections, memodict)
        new_cache.cache_client = self.cache_client
        return new_cache
