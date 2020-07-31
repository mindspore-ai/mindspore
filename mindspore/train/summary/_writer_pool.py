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
"""Write events to disk in a base directory."""
import os
import time
from collections import deque
from multiprocessing import Pool, Process, Queue, cpu_count

import mindspore.log as logger

from ._lineage_adapter import serialize_to_lineage_event
from ._summary_adapter import package_graph_event, package_summary_event
from ._summary_writer import LineageWriter, SummaryWriter


def _pack_data(datadict, wall_time):
    """Pack data according to which plugin."""
    result, summaries, step = [], [], None
    for plugin, datalist in datadict.items():
        for data in datalist:
            if plugin == 'graph':
                result.append([plugin, package_graph_event(data.get('value')).SerializeToString()])
            elif plugin in ('train_lineage', 'eval_lineage', 'custom_lineage_data', 'dataset_graph'):
                result.append([plugin, serialize_to_lineage_event(plugin, data.get('value'))])
            elif plugin in ('scalar', 'tensor', 'histogram', 'image'):
                summaries.append({'_type': plugin.title(), 'name': data.get('tag'), 'data': data.get('value')})
                step = data.get('step')
    if summaries:
        result.append(['summary', package_summary_event(summaries, step, wall_time).SerializeToString()])
    return result


class WriterPool(Process):
    """
    Use a set of pooled resident processes for writing a list of file.

    Args:
        base_dir (str): The base directory to hold all the files.
        filelist (str): The mapping from short name to long filename.
    """

    def __init__(self, base_dir, max_file_size, **filedict) -> None:
        super().__init__()
        self._base_dir, self._filedict = base_dir, filedict
        self._queue, self._writers_ = Queue(cpu_count() * 2), None
        self._max_file_size = max_file_size
        self.start()

    def run(self):
        with Pool(min(cpu_count(), 32)) as pool:
            deq = deque()
            while True:
                while deq and deq[0].ready():
                    for plugin, data in deq.popleft().get():
                        self._write(plugin, data)

                if not self._queue.empty():
                    action, data = self._queue.get()
                    if action == 'WRITE':
                        deq.append(pool.apply_async(_pack_data, (data, time.time())))
                    elif action == 'FLUSH':
                        self._flush()
                    elif action == 'END':
                        break
            for result in deq:
                for plugin, data in result.get():
                    self._write(plugin, data)

            self._close()

    @property
    def _writers(self):
        """Get the writers in the subprocess."""
        if self._writers_ is not None:
            return self._writers_
        self._writers_ = []
        for plugin, filename in self._filedict.items():
            filepath = os.path.join(self._base_dir, filename)
            if plugin == 'summary':
                self._writers_.append(SummaryWriter(filepath, self._max_file_size))
            elif plugin == 'lineage':
                self._writers_.append(LineageWriter(filepath, self._max_file_size))
        return self._writers_

    def _write(self, plugin, data):
        """Write the data in the subprocess."""
        for writer in self._writers[:]:
            try:
                writer.write(plugin, data)
            except RuntimeError as e:
                logger.warning(e.args[0])
                self._writers.remove(writer)
                writer.close()

    def _flush(self):
        """Flush the writers in the subprocess."""
        for writer in self._writers:
            writer.flush()

    def _close(self):
        """Close the writers in the subprocess."""
        for writer in self._writers:
            writer.close()

    def write(self, data) -> None:
        """
        Write the event to file.

        Args:
            name (str): The key of a specified file.
            data (Optional[str, Tuple[list, int]]): The data to write.
        """
        self._queue.put(('WRITE', data))

    def flush(self):
        """Flush the writer and sync data to disk."""
        self._queue.put(('FLUSH', None))

    def close(self) -> None:
        """Close the writer."""
        self._queue.put(('END', None))
        self.join()
