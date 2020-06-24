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
from collections import deque
from multiprocessing import Pool, Process, Queue, cpu_count

from ._lineage_adapter import serialize_to_lineage_event
from ._summary_adapter import package_graph_event, package_summary_event
from ._summary_writer import SummaryWriter, LineageWriter


def _pack_data(datadict):
    """Pack data according to which plugin."""
    result = []
    summaries, step, mode = [], None, None
    for plugin, datalist in datadict.items():
        for data in datalist:
            if plugin == 'graph':
                result.append([plugin, data.get('mode'), package_graph_event(data.get('value')).SerializeToString()])
            elif plugin in ('train_lineage', 'eval_lineage', 'custom_lineage_data', 'dataset_graph'):
                result.append([plugin, data.get('mode'), serialize_to_lineage_event(plugin, data.get('value'))])
            elif plugin in ('scalar', 'tensor', 'histogram', 'image'):
                summaries.append({'_type': plugin.title(), 'name': data.get('tag'), 'data': data.get('value')})
                step = data.get('step')
                mode = data.get('mode')
    if summaries:
        result.append(['summary', mode, package_summary_event(summaries, step).SerializeToString()])
    return result


class WriterPool(Process):
    """
    Use a set of pooled resident processes for writing a list of file.

    Args:
        base_dir (str): The base directory to hold all the files.
        filelist (str): The mapping from short name to long filename.
    """

    def __init__(self, base_dir, **filedict) -> None:
        super().__init__()
        self._base_dir, self._filedict = base_dir, filedict
        self._queue = Queue(cpu_count() * 2)
        self.start()

    def run(self):
        writers = self._get_writers()

        with Pool(min(cpu_count(), 32)) as pool:
            deq = deque()
            while True:
                while deq and deq[0].ready():
                    for plugin, mode, data in deq.popleft().get():
                        for writer in writers:
                            writer.write(plugin, mode, data)

                if not self._queue.empty():
                    action, data = self._queue.get()
                    if action == 'WRITE':
                        deq.append(pool.apply_async(_pack_data, (data,)))
                    elif action == 'FLUSH':
                        for writer in writers:
                            writer.flush()
                    elif action == 'END':
                        break
            for result in deq:
                for plugin, mode, data in result.get():
                    for writer in writers:
                        writer.write(plugin, mode, data)

            for writer in writers:
                writer.close()

    def _get_writers(self):
        writers = []
        for plugin, filename in self._filedict.items():
            filepath = os.path.join(self._base_dir, filename)
            if plugin == 'summary':
                writers.append(SummaryWriter(filepath))
            elif plugin == 'lineage':
                writers.append(LineageWriter(filepath))
        return writers

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
