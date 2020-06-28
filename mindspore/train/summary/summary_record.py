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
"""Record the summary event."""
import os
import re
import threading

from mindspore import log as logger

from ..._c_expression import Tensor
from ..._checkparam import _check_str_by_regular
from .._utils import _make_directory, _check_to_numpy, _check_lineage_value
from ._summary_adapter import get_event_file_name, package_graph_event
from ._writer_pool import WriterPool

# for the moment, this lock is for caution's sake,
# there are actually no any concurrencies happening.
_summary_lock = threading.Lock()
# cache the summary data
_summary_tensor_cache = {}


def _cache_summary_tensor_data(summary):
    """
    Get the time of ms.

    Args:
         summary (list): [{"name": tag_name, "data": tensor}, {"name": tag_name, "data": tensor},...].
    """
    with _summary_lock:
        for item in summary:
            _summary_tensor_cache[item['name']] = item['data']
        return True


def _get_summary_tensor_data():
    global _summary_tensor_cache
    with _summary_lock:
        data = _summary_tensor_cache
        _summary_tensor_cache = {}
        return data


def _dictlist():
    from collections import defaultdict
    return defaultdict(list)


class SummaryRecord:
    """
    SummaryRecord is used to record the summary data and lineage data.

    Note:
        The API will create a summary file and a lineage file lazily in a given directory and writes data to them.
        It writes the data to files by executing the record method. In addition to record the data bubbled up from
        the network by defining the summary operators, SummaryRecord also supports to record extra data which
        can be added by calling add_value. Finally, make sure to close the SummaryRecord object at the end.

    Args:
        log_dir (str): The log_dir is a directory location to save the summary.
        queue_max_size (int): The capacity of event queue.(reserved). Default: 0.
        flush_time (int): Frequency to flush the summaries to disk, the unit is second. Default: 120.
        file_prefix (str): The prefix of file. Default: "events".
        file_suffix (str): The suffix of file. Default: "_MS".
        network (Cell): Obtain a pipeline through network for saving graph summary. Default: None.

    Raises:
        TypeError: If `queue_max_size` and `flush_time` is not int, or `file_prefix` and `file_suffix` is not str.
        RuntimeError: If the log_dir can not be resolved to a canonicalized absolute pathname.

    Examples:
        >>> with SummaryRecord(log_dir="/opt/log", file_prefix="xxx_", file_suffix="_yyy") as summary_record:
        >>>     pass
    """

    def __init__(self,
                 log_dir,
                 queue_max_size=0,
                 flush_time=120,
                 file_prefix="events",
                 file_suffix="_MS",
                 network=None):

        self._closed, self._mode = False, 'train'
        self._data_pool = _dictlist()

        _check_str_by_regular(file_prefix)
        _check_str_by_regular(file_suffix)

        self.log_path = _make_directory(log_dir)

        if not isinstance(queue_max_size, int) or not isinstance(flush_time, int):
            raise TypeError("`queue_max_size` and `flush_time` should be int")
        if not isinstance(file_prefix, str) or not isinstance(file_suffix, str):
            raise TypeError("`file_prefix` and `file_suffix`  should be str.")

        self.queue_max_size = queue_max_size
        if queue_max_size < 0:
            # 0 is not limit
            logger.warning("The queue_max_size(%r) set error, will use the default value: 0", queue_max_size)
            self.queue_max_size = 0

        self.flush_time = flush_time
        if flush_time <= 0:
            logger.warning("The flush_time(%r) set error, will use the default value: 120", flush_time)
            self.flush_time = 120

        self.prefix = file_prefix
        self.suffix = file_suffix
        self.network = network
        self.has_graph = False

        # create the summary writer file
        self.event_file_name = get_event_file_name(self.prefix, self.suffix)
        try:
            self.full_file_name = os.path.join(self.log_path, self.event_file_name)
        except Exception as ex:
            raise RuntimeError(ex)

        self._event_writer = WriterPool(log_dir,
                                        summary=self.full_file_name,
                                        lineage=get_event_file_name('events', '_lineage'))

    def __enter__(self):
        """Enter the context manager."""
        if self._closed:
            raise ValueError('SummaryRecord has been closed.')
        return self

    def __exit__(self, extype, exvalue, traceback):
        """Exit the context manager."""
        self.close()

    def set_mode(self, mode):
        """
        Set the mode for the recorder to be aware. The mode is set 'train' by default.

        Args:
            mode (str): The mode to set, which should be 'train' or 'eval'.

        Raises:
            ValueError: When the mode is not recognized.

        Examples:
            >>> with SummaryRecord(log_dir="/opt/log", file_prefix="xxx_", file_suffix="_yyy") as summary_record:
            >>>     summary_record.set_mode('eval')
        """
        mode_spec = 'train', 'eval'
        if mode not in mode_spec:
            raise ValueError(f'{repr(mode)} is not a recognized mode.')
        self._mode = mode

    def add_value(self, plugin, name, value):
        """
        Add value to be record later on.

        When the plugin is 'tensor', 'scalar', 'image' or 'histogram',
        the name should be the tag name, and the value should be a Tensor.

        When the plugin plugin is 'graph', the value should be a GraphProto.

        When the plugin 'dataset_graph', 'train_lineage', 'eval_lineage',
        or 'custom_lineage_data', the value should be a proto message.


        Args:
            plugin (str): The plugin for the value.
            name (str): The name for the value.
            value (Union[Tensor, GraphProto, TrainLineage, EvaluationLineage, DatasetGraph, UserDefinedInfo]): \
                The value to store.

                - GraphProto: The 'value' should be a serialized string this type when the plugin is 'graph'.
                - Tensor: The 'value' should be this type when the plugin is 'scalar', 'image', 'tensor' or 'histogram'.
                - TrainLineage: The 'value' should be this type when the plugin is 'train_lineage'.
                - EvaluationLineage: The 'value' should be this type when the plugin is 'eval_lineage'.
                - DatasetGraph: The 'value' should be this type when the plugin is 'dataset_graph'.
                - UserDefinedInfo: The 'value' should be this type when the plugin is 'custom_lineage_data'.

        Raises:
            ValueError: When the name is not valid.
            TypeError: When the value is not a Tensor.

        Examples:
            >>> with SummaryRecord(log_dir="/opt/log", file_prefix="xxx_", file_suffix="_yyy") as summary_record:
            >>>     summary_record.add_value('scalar', 'loss', Tensor(0.1))
        """
        if plugin in ('tensor', 'scalar', 'image', 'histogram'):
            if not name or not isinstance(name, str):
                raise ValueError(f'{repr(name)} is not a valid tag name.')
            if not isinstance(value, Tensor):
                raise TypeError(f'Expect the value to be Tensor, but got {type(value).__name__}')
            np_value = _check_to_numpy(plugin, value)
            if name in {item['tag'] for item in self._data_pool[plugin]}:
                entry = repr(f'{name}/{plugin}')
                logger.warning(f'{entry} has duplicate values. Only the newest one will be recorded.')
            self._data_pool[plugin].append(dict(tag=name, mode=self._mode, value=np_value))

        elif plugin in ('train_lineage', 'eval_lineage', 'dataset_graph', 'custom_lineage_data'):
            _check_lineage_value(plugin, value)
            self._data_pool[plugin].append(dict(mode=self._mode, value=value.SerializeToString()))
        elif plugin == 'graph':
            package_graph_event(value)
            self._data_pool[plugin].append(dict(mode=self._mode, value=value))
        else:
            raise ValueError(f'No such plugin of {repr(plugin)}')

    def record(self, step, train_network=None):
        """
        Record the summary.

        Args:
            step (int): Represents training step number.
            train_network (Cell): The network that called the callback.

        Returns:
            bool, whether the record process is successful or not.

        Examples:
            >>> with SummaryRecord(log_dir="/opt/log", file_prefix="xxx_", file_suffix="_yyy") as summary_record:
            >>>     summary_record.record(step=2)
        """
        logger.info("SummaryRecord step is %r.", step)
        if self._closed:
            logger.error("The record writer is closed.")
            return False
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError("`step` should be int")
        # Set the current summary of train step
        if self.network is not None and not self.has_graph:
            graph_proto = self.network.get_func_graph_proto()
            if graph_proto is None and train_network is not None:
                graph_proto = train_network.get_func_graph_proto()
            if graph_proto is None:
                logger.error("Failed to get proto for graph")
            else:
                self._event_writer.write({'graph': [{'step': step, 'value': graph_proto}]})
                self.has_graph = True
                if not _summary_tensor_cache:
                    return True

        if self._mode == 'train':
            self._add_summary_tensor_data()

        self._event_writer.write(self._consume_data_pool(step))
        return True

    def _add_summary_tensor_data(self):
        summary_data = _get_summary_tensor_data()
        if not summary_data:
            logger.debug(f'No summary data bubbled from the network.')
        for name, tensor in summary_data.items():
            tag, plugin = SummaryRecord._parse_from(name)
            if (tag, plugin) == (None, None):
                logger.warning("The name(%r) is invalid, expected 'TAG[:TYPE]'.", name)
            else:
                self.add_value(plugin.lower(), tag, tensor)

    def _consume_data_pool(self, step):
        try:
            for values in self._data_pool.values():
                for value in values:
                    value['step'] = step
            return self._data_pool
        finally:
            self._data_pool = _dictlist()

    @property
    def log_dir(self):
        """
        Get the full path of the log file.

        Returns:
            str, the full path of log file.

        Examples:
            >>> with SummaryRecord(log_dir="/opt/log", file_prefix="xxx_", file_suffix="_yyy") as summary_record:
            >>>     print(summary_record.log_dir)
        """
        return self.full_file_name

    def flush(self):
        """
        Flush the event file to disk.

        Call it to make sure that all pending events have been written to disk.

        Examples:
            >>> with SummaryRecord(log_dir="/opt/log", file_prefix="xxx_", file_suffix="_yyy") as summary_record:
            >>>     summary_record.flush()
        """
        if self._closed:
            logger.error("The record writer is closed and can not flush.")
        elif self._event_writer:
            self._event_writer.flush()

    def close(self):
        """
        Flush all events and close summary records. Please use with statement to autoclose.

        Examples:
            >>> with SummaryRecord(log_dir="/opt/log", file_prefix="xxx_", file_suffix="_yyy") as summary_record:
            >>>     pass # summary_record autoclosed
        """
        if not self._closed and self._event_writer:
            # event writer flush and close
            logger.info('Please wait it may take quite some time to finish writing and closing.')
            self._event_writer.close()
            self._closed = True

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _parse_from(name: str = None):
        """Parse the tag and type from name."""
        if not isinstance(name, str):
            return None, None
        match = re.match(r'(.+)\[:(.+)\]', name)
        if match:
            return match.groups()
        return None, None
