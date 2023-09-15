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

from ..core.validator_helpers import type_check, check_pos_int32, check_pos_uint32, check_uint64, check_positive, \
    check_value


class DatasetCache:
    """
    A client to interface with tensor caching service.

    For details, please check `Tutorial <https://www.mindspore.cn/
    tutorials/experts/en/master/dataset/cache.html>`_ .

    Args:
        session_id (int): A user assigned session id for the current pipeline.
        size (int, optional): Size of the memory set aside for the row caching. Default: ``0``, which means unlimited,
            note that it might bring in the risk of running out of memory on the machine.
        spilling (bool, optional): Whether or not spilling to disk if out of memory. Default: ``False``.
        hostname (str, optional): Host name. Default: ``None`` , use default hostname '127.0.0.1'.
        port (int, optional): Port to connect to server. Default: ``None`` , use default port 50052.
        num_connections (int, optional): Number of tcp/ip connections. Default: ``None`` , use default value 12.
        prefetch_size (int, optional): The size of the cache queue between operations.
            Default: ``None`` , use default value 20.

    Examples:
        >>> import subprocess
        >>> import mindspore.dataset as ds
        >>>
        >>> # Create a cache instance with command line `cache_admin --start` and create a session with `cache_admin -g`
        >>> # After creating cache with a valid session, get session id with command `cache_admin --list_sessions`
        >>> session_id = subprocess.getoutput('cache_admin --list_sessions | tail -1 | awk -F " " \'{{print $1;}}\'')
        >>> some_cache = ds.DatasetCache(session_id=int(session_id), size=0)
        >>>
        >>> dataset_dir = "/path/to/image_folder_dataset_directory"
        >>> dataset = ds.ImageFolderDataset(dataset_dir, cache=some_cache)
    """

    def __init__(self, session_id, size=0, spilling=False, hostname=None, port=None, num_connections=None,
                 prefetch_size=None):
        check_pos_uint32(session_id, "session_id")
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
            check_pos_int32(num_connections, "num_connections")
        if prefetch_size is not None:
            check_pos_int32(prefetch_size, "prefetch_size")

        self.session_id = session_id
        self.size = size
        self.spilling = spilling
        self.hostname = hostname
        self.port = port
        self.prefetch_size = prefetch_size
        self.num_connections = num_connections
        self.cache_client = CacheClient(session_id, size, spilling, hostname, port, num_connections, prefetch_size)

    def get_stat(self):
        """
        Get the statistics from a cache. After data pipeline, three types of statistics can be obtained,
        including average number of cache hits (avg_cache_sz), number of caches in memory (num_mem_cached)
        and number of caches in disk (num_disk_cached).

        Examples:
            >>> import os
            >>> import mindspore.dataset as ds
            >>>
            >>> # In example above, we created cache with a valid session id
            >>> id = int(os.popen('cache_admin --list_sessions | tail -1 | awk -F " " \'{{print $1;}}\'').read())
            >>> some_cache = ds.DatasetCache(session_id=id, size=0)
            >>>
            >>> # run the dataset pipeline to trigger cache
            >>> dataset = ds.ImageFolderDataset("/path/to/image_folder_dataset_directory", cache=some_cache)
            >>> data = list(dataset)
            >>>
            >>> # get status of cache
            >>> stat = some_cache.get_stat()
            >>> # Average cache size
            >>> cache_sz = stat.avg_cache_sz
            >>> # Number of rows cached in memory
            >>> num_mem_cached = stat.num_mem_cached
            >>> # Number of rows spilled to disk
            >>> num_disk_cached = stat.num_disk_cached
        """
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
