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
"""Writes events to disk in a logdir."""
import os
import stat
from collections import deque
from multiprocessing import Pool, Process, Queue, cpu_count

from ..._c_expression import EventWriter_
from ._summary_adapter import package_summary_event


def _pack(result, step):
    summary_event = package_summary_event(result, step)
    return summary_event.SerializeToString()


class EventWriter(Process):
    """
    Creates a `EventWriter` and write event to file.

    Args:
        filepath (str): Summary event file path and file name.
        flush_interval (int): The flush seconds to flush the pending events to disk. Default: 120.
    """

    def __init__(self, filepath: str, flush_interval: int) -> None:
        super().__init__()
        _ = flush_interval
        with open(filepath, 'w'):
            os.chmod(filepath, stat.S_IWUSR | stat.S_IRUSR)
        self._writer = EventWriter_(filepath)
        self._queue = Queue(cpu_count() * 2)
        self.start()

    def run(self):

        with Pool() as pool:
            deq = deque()
            while True:
                while deq and deq[0].ready():
                    self._writer.Write(deq.popleft().get())

                if not self._queue.empty():
                    action, data = self._queue.get()
                    if action == 'WRITE':
                        if not isinstance(data, (str, bytes)):
                            deq.append(pool.apply_async(_pack, data))
                        else:
                            self._writer.Write(data)
                    elif action == 'FLUSH':
                        self._writer.Flush()
                    elif action == 'END':
                        break
            for res in deq:
                self._writer.Write(res.get())

            self._writer.Shut()

    def write(self, data) -> None:
        """
        Write the event to file.

        Args:
            data (Optional[str, Tuple[list, int]]): The data to write.
        """
        self._queue.put(('WRITE', data))

    def flush(self):
        """Flush the writer."""
        self._queue.put(('FLUSH', None))

    def close(self) -> None:
        """Close the writer."""
        self._queue.put(('END', None))
        self.join()
