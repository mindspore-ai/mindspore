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
# ==============================================================================
"""
This dataset module creates an internal queue class to more optimally pass data
between multiple processes in Python.  It has same API as multiprocessing.queue
but it will pass large data through shared memory.
"""

import multiprocessing.queues
import multiprocessing
import types
import numpy as np

from mindspore import log as logger
from ..transforms.py_transforms_util import ExceptionHandler


class _SharedQueue(multiprocessing.queues.Queue):
    """
    Class to implement a queue using shared memory for better performance.
    Args:
        size: Number of elements in the queue.
        copy_out: Flag to indidcate whether an extra copy should be done before returning.  If data will immediately be
                  copied before returning, then this can be set to False.
        max_rowsize: Maximum size of any element in the Queue in MB.
    """

    def __init__(self, size, copy_out=False, max_rowsize=6):
        super().__init__(size, ctx=multiprocessing.get_context())

        self.copy_out = copy_out

        # change max_rowsize in MB into bytes
        self.seg_size = max_rowsize * 1024 * 1024
        ##pipe can hold up to 65,636 bytes at a time
        self.min_shared_mem = 10000
        self.shm_list = []
        self.seg_pos = 0
        # num_seg has to be 2 more than the queue size.  We can have remote worker filling a buffer, main process
        # reading a buffer and also have a full queue of buffers in the meta-data queue
        self.num_seg = size + 2
        self.data_immediate = 0
        self.data_shared = 1
        self.print_error = True

        try:
            for _ in range(self.num_seg):
                a = multiprocessing.Array("b", self.seg_size)
                self.shm_list.append(a)
        except Exception:
            raise RuntimeError(
                "_SharedQueue: Error allocating "
                + str(self.seg_size)
                + "bytes, "
                + str(self.num_seg)
                + " elements."
                + " This might be caused by insufficient shm, and the recommended shm size is at least 5 GB."
            )

    def put(self, data, timeout=None):
        if isinstance(data, ExceptionHandler):
            super().put(data, timeout=timeout)
        else:
            name_list = []
            count = 0
            start_bytes = 0
            if not isinstance(data, tuple) and not isinstance(data, np.ndarray):
                raise TypeError("return value of user defined python function in GeneratorDataset or"
                                " map should be numpy array or tuple of numpy array.")
            for r in data:
                # the map:pyfunc is a yield generator which can't be serialize
                if isinstance(r, types.GeneratorType):
                    raise TypeError("Can not pickle {} object, please verify pyfunc return with numpy array"
                                    .format(type(r)))
                if (isinstance(r, np.ndarray) and r.size > self.min_shared_mem
                        and start_bytes + r.nbytes < self.seg_size):
                    # need to convert start_bytes to offset in array
                    start_offset = start_bytes
                    dest = np.ndarray(r.shape, r.dtype, buffer=self.shm_list[self.seg_pos].get_obj(),
                                      offset=start_offset)
                    np.copyto(dest, r)
                    byte = r.nbytes
                    byte = 8 * ((byte + 7) // 8)
                    start_bytes += byte
                    name_list.append((self.data_shared, self.seg_pos, byte, r.dtype, r.shape))
                    count += 1
                else:
                    if isinstance(r, np.ndarray) and r.size >= self.min_shared_mem:
                        # Only print out error the first time it happens
                        if self.print_error:
                            logger.warning(
                                "Using shared memory queue, but rowsize is larger than allocated memory "
                                + "max_rowsize "
                                + str(self.seg_size)
                                + " current rowsize "
                                + str(start_bytes + r.nbytes)
                            )
                            self.print_error = False
                    name_list.append((self.data_immediate, r))
            super().put(name_list, timeout=timeout)
            # note above could generate a queue full exception.  It will be handled by teh caller
            # only increment seg_pos after successfully adding to metadata queue

            if start_bytes > 0:
                self.seg_pos = (self.seg_pos + 1) % self.num_seg

    def get(self, timeout=None):
        result = super().get(timeout=timeout)
        if isinstance(result, ExceptionHandler):
            return result
        r = []
        start_bytes = 0
        for x in result:
            if x[0] == self.data_shared:
                seg_pos = x[1]
                byte = x[2]
                dtype = x[3]
                shape = x[4]
                start_offset = start_bytes
                b = self.shm_list[seg_pos]
                data = np.ndarray(shape, dtype, buffer=b.get_obj(), offset=start_offset)
                start_bytes += byte
                if self.copy_out:
                    data2 = np.copy(data)
                    r.append(data2)
                else:
                    r.append(data)
            elif x[0] == self.data_immediate:
                r.append(x[1])
            else:
                raise RuntimeError("SharedQueue, invalid entry in metadata.")
        return tuple(r)
