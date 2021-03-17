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
"""Record the summary event."""
import atexit
import os
import re
import threading
import time
from collections import defaultdict

from mindspore import log as logger
from mindspore.nn import Cell

from ..._c_expression import Tensor
from ..._checkparam import Validator
from .._utils import _check_lineage_value, _check_to_numpy, _make_directory, check_value_type
from ._summary_adapter import get_event_file_name, package_graph_event
from ._explain_adapter import check_explain_proto
from ._writer_pool import WriterPool

# for the moment, this lock is for caution's sake,
# there are actually no any concurrences happening.
_summary_lock = threading.Lock()
# cache the summary data
_summary_tensor_cache = {}
_DEFAULT_EXPORT_OPTIONS = {
    'tensor_format': {'npy', None},
}


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


def process_export_options(export_options):
    """Check specified data type and value."""
    if export_options is None:
        return None

    check_value_type('export_options', export_options, [dict, type(None)])

    for export_option, export_format in export_options.items():
        check_value_type('export_option', export_option, [str])
        check_value_type('export_format', export_format, [str, type(None)])

    unexpected_params = set(export_options) - set(_DEFAULT_EXPORT_OPTIONS)
    if unexpected_params:
        raise ValueError(f'For `export_options` the keys {unexpected_params} are unsupported, '
                         f'expect the follow keys: {list(_DEFAULT_EXPORT_OPTIONS.keys())}')

    for export_option, export_format in export_options.items():
        unexpected_format = {export_format} - _DEFAULT_EXPORT_OPTIONS.get(export_option)
        if unexpected_format:
            raise ValueError(
                f'For `export_options`, the export_format {unexpected_format} are unsupported for {export_option}, '
                f'expect the follow values: {list(_DEFAULT_EXPORT_OPTIONS.get(export_option))}')

    for item in set(export_options):
        check_value_type(item, export_options.get(item), [str, type(None)])

    return export_options


class SummaryRecord:
    """
    SummaryRecord is used to record the summary data and lineage data.

    The API will create a summary file and lineage files lazily in a given directory and writes data to them.
    It writes the data to files by executing the 'record' method. In addition to recording the data bubbled up from
    the network by defining the summary operators, SummaryRecord also supports to record extra data which
    can be added by calling add_value.

    Note:
        1. Make sure to close the SummaryRecord at the end, otherwise the process will not exit.
           Please see the Example section below to learn how to close properly in two ways.
        2. Only one SummaryRecord instance is allowed at a time, otherwise it will cause data writing problems.
        3. SummaryRecord only supports Linux systems.

    Args:
        log_dir (str): The log_dir is a directory location to save the summary.
        file_prefix (str): The prefix of file. Default: "events".
        file_suffix (str): The suffix of file. Default: "_MS".
        network (Cell): Obtain a pipeline through network for saving graph summary. Default: None.
        max_file_size (int, optional): The maximum size of each file that can be written to disk (in bytes).
            Unlimited by default. For example, to write not larger than 4GB, specify `max_file_size=4 * 1024 ** 3`.
        raise_exception (bool, optional): Sets whether to throw an exception when a RuntimeError or OSError exception
            occurs in recording data. Default: False, this means that error logs are printed and no exception is thrown.
        export_options (Union[None, dict]): Perform custom operations on the export data.
            Note that the size of export files is not limited by the max_file_size.
            You can customize the export data with a dictionary. For example, you can set {'tensor_format': 'npy'}
            to export tensor as npy file. The data that supports control is shown below. Default: None, it means that
            the data is not exported.

            - tensor_format (Union[str, None]): Customize the export tensor format. Supports ["npy", None].
              Default: None, it means that the tensor is not exported.

              - npy: export tensor as npy file.

    Raises:
        TypeError: If the parameter type is incorrect.

    Examples:
        >>> from mindspore.train.summary import SummaryRecord
        >>> if __name__ == '__main__':
        ...     # use in with statement to auto close
        ...     with SummaryRecord(log_dir="./summary_dir") as summary_record:
        ...         pass
        ...
        ...     # use in try .. finally .. to ensure closing
        ...     try:
        ...         summary_record = SummaryRecord(log_dir="./summary_dir")
        ...     finally:
        ...         summary_record.close()
    """

    def __init__(self, log_dir, file_prefix="events", file_suffix="_MS",
                 network=None, max_file_size=None, raise_exception=False, export_options=None):

        self._closed, self._event_writer = False, None
        self._mode, self._data_pool = 'train', defaultdict(list)

        Validator.check_str_by_regular(file_prefix)
        Validator.check_str_by_regular(file_suffix)

        self.log_path = _make_directory(log_dir)

        if not isinstance(max_file_size, (int, type(None))):
            raise TypeError("The 'max_file_size' should be int type.")

        if not isinstance(file_prefix, str) or not isinstance(file_suffix, str):
            raise TypeError("`file_prefix` and `file_suffix`  should be str.")

        if max_file_size is not None and max_file_size < 0:
            logger.warning("The 'max_file_size' should be greater than 0.")
            max_file_size = None

        Validator.check_value_type(arg_name='raise_exception', arg_value=raise_exception, valid_types=bool)

        self.prefix = file_prefix
        self.suffix = file_suffix
        self.network = network
        self.has_graph = False

        time_second = str(int(time.time()))
        # create the summary writer file
        self.event_file_name = get_event_file_name(self.prefix, self.suffix, time_second)
        self.full_file_name = os.path.join(self.log_path, self.event_file_name)

        self._export_options = process_export_options(export_options)
        export_dir = ''
        if self._export_options is not None:
            export_dir = "export_{}".format(time_second)

        filename_dict = dict(summary=self.event_file_name,
                             lineage=get_event_file_name(self.prefix, '_lineage', time_second),
                             explainer=get_event_file_name(self.prefix, '_explain', time_second),
                             exporter=export_dir)
        self._event_writer = WriterPool(log_dir,
                                        max_file_size,
                                        raise_exception,
                                        **filename_dict)
        _get_summary_tensor_data()
        atexit.register(self.close)

    def __enter__(self):
        """Enter the context manager."""
        if self._closed:
            raise ValueError('SummaryRecord has been closed.')
        return self

    def __exit__(self, *err):
        """Exit the context manager."""
        self.close()

    def set_mode(self, mode):
        """
        Set the training phase. Different training phases affect data recording.

        Args:
            mode (str): The mode to be set, which should be 'train' or 'eval'. When the mode is 'eval',
                summary_record will not record the data of summary operators.

        Raises:
            ValueError: When the mode is not recognized.

        Examples:
            >>> from mindspore.train.summary import SummaryRecord
            >>> if __name__ == '__main__':
            ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
            ...         summary_record.set_mode('eval')
        """
        mode_spec = 'train', 'eval'
        if mode not in mode_spec:
            raise ValueError(f'{repr(mode)} is not a recognized mode.')
        self._mode = mode

    def add_value(self, plugin, name, value):
        """
        Add value to be recorded later.

        Args:
            plugin (str): The value of the plugin.
            name (str): The value of the name.
            value (Union[Tensor, GraphProto, TrainLineage, EvaluationLineage, DatasetGraph, UserDefinedInfo]): \
                The value to store.

                - The data type of value should be 'GraphProto' (see mindspore/ccsrc/anf_ir.proto) object
                  when the plugin is 'graph'.
                - The data type of value should be 'Tensor' object when the plugin is 'scalar', 'image', 'tensor'
                  or 'histogram'.
                - The data type of value should be a 'TrainLineage' object when the plugin is 'train_lineage',
                  see mindspore/ccsrc/lineage.proto.
                - The data type of value should be a 'EvaluationLineage' object when the plugin is 'eval_lineage',
                  see mindspore/ccsrc/lineage.proto.
                - The data type of value should be a 'DatasetGraph' object when the plugin is 'dataset_graph',
                  see mindspore/ccsrc/lineage.proto.
                - The data type of value should be a 'UserDefinedInfo' object when the plugin is 'custom_lineage_data',
                  see mindspore/ccsrc/lineage.proto.
                - The data type of value should be a 'Explain' object when the plugin is 'explainer',
                  see mindspore/ccsrc/summary.proto.
        Raises:
            ValueError: If the parameter value is invalid.
            TypeError: If the parameter type is error.

        Examples:
            >>> from mindspore import Tensor
            >>> from mindspore.train.summary import SummaryRecord
            >>> if __name__ == '__main__':
            ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
            ...         summary_record.add_value('scalar', 'loss', Tensor(0.1))
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
            data = dict(tag=name, value=np_value)
            export_plugin = '{}_format'.format(plugin)
            if self._export_options is not None and export_plugin in self._export_options:
                data['export_option'] = self._export_options.get(export_plugin)
            self._data_pool[plugin].append(data)

        elif plugin in ('train_lineage', 'eval_lineage', 'dataset_graph', 'custom_lineage_data'):
            _check_lineage_value(plugin, value)
            self._data_pool[plugin].append(dict(value=value.SerializeToString()))
        elif plugin == 'graph':
            package_graph_event(value)
            self._data_pool[plugin].append(dict(value=value))
        elif plugin == 'explainer':
            check_explain_proto(value)
            self._data_pool[plugin].append(dict(value=value.SerializeToString()))
        else:
            raise ValueError(f'No such plugin of {repr(plugin)}')

    def record(self, step, train_network=None, plugin_filter=None):
        """
        Record the summary.

        Args:
            step (int): Represents training step number.
            train_network (Cell): The network to call the callback.
            plugin_filter (Optional[Callable[[str], bool]]): The filter function, \
                which is used to filter out plugins from being written by returning False.

        Returns:
            bool, whether the record process is successful or not.

        Raises:
            TypeError: If the parameter type is error.
            RuntimeError: If the disk space is insufficient.

        Examples:
            >>> from mindspore.train.summary import SummaryRecord
            >>> if __name__ == '__main__':
            ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
            ...         summary_record.record(step=2)
            ...
            True
        """
        logger.debug("SummaryRecord step is %r.", step)
        Validator.check_value_type(arg_name='step', arg_value=step, valid_types=int)
        Validator.check_value_type(arg_name='train_network', arg_value=train_network, valid_types=[Cell, type(None)])

        if self._closed:
            logger.error("The record writer is closed.")
            return False
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

        if not plugin_filter:
            self._event_writer.write(self._consume_data_pool(step))
        else:
            filtered = {}
            for plugin, datalist in self._consume_data_pool(step).items():
                if plugin_filter(plugin):
                    filtered[plugin] = datalist
            self._event_writer.write(filtered)
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
            self._data_pool = defaultdict(list)

    @property
    def log_dir(self):
        """
        Get the full path of the log file.

        Returns:
            str, the full path of log file.

        Examples:
            >>> from mindspore.train.summary import SummaryRecord
            >>> if __name__ == '__main__':
            ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
            ...         log_dir = summary_record.log_dir
        """
        return self.full_file_name

    def flush(self):
        """
        Flush the event file to disk.

        Call it to make sure that all pending events have been written to disk.

        Examples:
            >>> from mindspore.train.summary import SummaryRecord
            >>> if __name__ == '__main__':
            ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
            ...         summary_record.flush()
        """
        if self._closed:
            logger.error("The record writer is closed and can not flush.")
        elif self._event_writer:
            self._event_writer.flush()

    def close(self):
        """
        Flush all events and close summary records. Please use the statement to autoclose.

        Examples:
            >>> from mindspore.train.summary import SummaryRecord
            >>> if __name__ == '__main__':
            ...     try:
            ...         summary_record = SummaryRecord(log_dir="./summary_dir")
            ...     finally:
            ...         summary_record.close()
        """
        if not self._closed and self._event_writer:
            # event writer flush and close
            logger.info('Please wait it may take quite some time to finish writing and closing.')
            atexit.unregister(self.close)
            self._event_writer.close()
            self._event_writer.join()
            self._closed = True

    @staticmethod
    def _parse_from(name: str = None):
        """Parse the tag and type from name."""
        if not isinstance(name, str):
            return None, None
        match = re.match(r'(.+)\[:(.+)\]', name)
        if match:
            return match.groups()
        return None, None
