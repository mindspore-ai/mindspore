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
import signal
from collections import deque

import mindspore.log as logger

from ._lineage_adapter import serialize_to_lineage_event
from ._summary_adapter import package_graph_event, package_summary_event
from ._explain_adapter import package_explain_event
from .writer import LineageWriter, SummaryWriter, ExplainWriter

try:
    from multiprocessing import get_context
    ctx = get_context('forkserver')
except ValueError:
    import multiprocessing as ctx


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
            elif plugin == 'explainer':
                result.append([plugin, package_explain_event(data.get('value'))])
    if summaries:
        result.append(['summary', package_summary_event(summaries, step, wall_time).SerializeToString()])
    return result


class WriterPool(ctx.Process):
    """
    Use a set of pooled resident processes for writing a list of file.

    Args:
        base_dir (str): The base directory to hold all the files.
        max_file_size (Optional[int]): The maximum size of each file that can be written to disk in bytes.
        filedict (dict): The mapping from plugin to filename.
    """

    def __init__(self, base_dir, max_file_size, **filedict) -> None:
        super().__init__()
        self._base_dir, self._filedict = base_dir, filedict
        self._queue, self._writers_ = ctx.Queue(ctx.cpu_count() * 2), None
        self._max_file_size = max_file_size
        self.start()

    def run(self):
        # Environment variables are used to specify a maximum number of OpenBLAS threads:
        # In ubuntu(GPU) environment, numpy will use too many threads for computing,
        # it may affect the start of the summary process.
        # Notice: At present, the performance of setting the thread to 2 has been tested to be more suitable.
        # If it is to be adjusted, it is recommended to test according to the scenario first
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        os.environ['GOTO_NUM_THREADS'] = '2'
        os.environ['OMP_NUM_THREADS'] = '2'

        # Prevent the multiprocess from capturing KeyboardInterrupt,
        # which causes the main process to fail to exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        with ctx.Pool(min(ctx.cpu_count(), 32)) as pool:
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
            elif plugin == 'explainer':
                self._writers_.append(ExplainWriter(filepath, self._max_file_size))
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
