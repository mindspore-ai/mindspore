# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import queue
from collections import deque

import psutil

import mindspore.log as logger
from mindspore.train.summary.enums import PluginEnum, WriterPluginEnum

from ._lineage_adapter import serialize_to_lineage_event
from ._summary_adapter import package_graph_event, package_summary_event
from ._explain_adapter import package_explain_event
from .writer import LineageWriter, SummaryWriter, ExplainWriter, ExportWriter

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
            if plugin == PluginEnum.GRAPH.value:
                result.append([plugin, package_graph_event(data.get('value')).SerializeToString()])
            elif plugin in (PluginEnum.TRAIN_LINEAGE.value, PluginEnum.EVAL_LINEAGE.value,
                            PluginEnum.CUSTOM_LINEAGE_DATA.value, PluginEnum.DATASET_GRAPH.value):
                result.append([plugin, serialize_to_lineage_event(plugin, data.get('value'))])
            elif plugin in (PluginEnum.SCALAR.value, PluginEnum.TENSOR.value, PluginEnum.HISTOGRAM.value,
                            PluginEnum.IMAGE.value):
                summaries.append({'_type': plugin.title(), 'name': data.get('tag'), 'data': data.get('value')})
                step = data.get('step')
            elif plugin == PluginEnum.EXPLAINER.value:
                result.append([plugin, package_explain_event(data.get('value'))])

            if 'export_option' in data:
                result.append([WriterPluginEnum.EXPORTER.value, data])

    if summaries:
        result.append(
            [WriterPluginEnum.SUMMARY.value, package_summary_event(summaries, step, wall_time).SerializeToString()])
    return result


class WriterPool(ctx.Process):
    """
    Use a set of pooled resident processes for writing a list of file.

    Args:
        base_dir (str): The base directory to hold all the files.
        max_file_size (Optional[int]): The maximum size of each file that can be written to disk in bytes.
        raise_exception (bool, optional): Sets whether to throw an exception when an RuntimeError exception occurs
            in recording data. Default: False, this means that error logs are printed and no exception is thrown.
        export_options (Union[None, dict]): Perform custom operations on the export data. Default: None.
        filedict (dict): The mapping from plugin to filename.
    """

    def __init__(self, base_dir, max_file_size, raise_exception=False, **filedict) -> None:
        super().__init__()
        self._base_dir, self._filedict = base_dir, filedict
        self._queue, self._writers_ = ctx.Queue(ctx.cpu_count() * 2), None
        self._max_file_size = max_file_size
        self._raise_exception = raise_exception
        self._training_pid = os.getpid()
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
                if self._check_heartbeat():
                    self._close()
                    return

                while deq and deq[0].ready():
                    for plugin, data in deq.popleft().get():
                        self._write(plugin, data)

                try:
                    action, data = self._queue.get(block=False)
                    if action == 'WRITE':
                        deq.append(pool.apply_async(_pack_data, (data, time.time())))
                    elif action == 'FLUSH':
                        self._flush()
                    elif action == 'END':
                        break
                except queue.Empty:
                    pass

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
            if plugin == WriterPluginEnum.SUMMARY.value:
                self._writers_.append(SummaryWriter(filepath, self._max_file_size))
            elif plugin == WriterPluginEnum.LINEAGE.value:
                self._writers_.append(LineageWriter(filepath, self._max_file_size))
            elif plugin == WriterPluginEnum.EXPLAINER.value:
                self._writers_.append(ExplainWriter(filepath, self._max_file_size))
            elif plugin == WriterPluginEnum.EXPORTER.value:
                self._writers_.append(ExportWriter(filepath, self._max_file_size))
        return self._writers_

    def _write(self, plugin, data):
        """Write the data in the subprocess."""
        for writer in self._writers[:]:
            try:
                writer.write(plugin, data)
            except (RuntimeError, OSError) as exc:
                logger.error(str(exc))
                self._writers.remove(writer)
                writer.close()
                if self._raise_exception:
                    raise
            except RuntimeWarning as exc:
                logger.warning(str(exc))
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
        super().close()

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

    def _check_heartbeat(self):
        """Check if the summary process should survive."""
        is_exit = False
        if not psutil.pid_exists(self._training_pid):
            logger.warning("The training process %d has exited, summary process will exit.", self._training_pid)
            is_exit = True

        if not self._writers:
            logger.warning("Can not find any writer to write summary data, "
                           "so SummaryRecord will not record data.")
            is_exit = True

        return is_exit
