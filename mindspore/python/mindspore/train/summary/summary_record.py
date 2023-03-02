# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

import atexit
import os
import re
import threading
import time
from collections import defaultdict

from mindspore import log as logger
from mindspore.nn import Cell
from mindspore import context
from mindspore._c_expression import security
from mindspore._c_expression import Tensor as Tensor_
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator
from mindspore.common.api import _cell_graph_executor
from mindspore.train._utils import _check_lineage_value, _check_to_numpy, _make_directory, check_value_type
from mindspore.train.summary._summary_adapter import get_event_file_name, package_graph_event
from mindspore.train.summary._writer_pool import WriterPool
from mindspore.train.summary.enums import PluginEnum
from mindspore.ops.operations import debug_ops

# for the moment, this lock is for caution's sake,
# there are actually no any concurrences happening.
_summary_lock = threading.Lock()
# cache the summary data
SUMMARY_TENSOR_CACHE = {}
_DEFAULT_EXPORT_OPTIONS = {
    'tensor_format': {'npy', None},
}
# Instance lock for counting SummaryRecord instance
_instance_lock = threading.Lock()


def _cache_summary_tensor_data(summary):
    """
    Get the time of ms.

    Args:
         summary (list): [{"name": tag_name, "data": tensor}, {"name": tag_name, "data": tensor},...].
    """
    with _summary_lock:
        for item in summary:
            SUMMARY_TENSOR_CACHE[item['name']] = item['data']
        return True


def _get_summary_tensor_data():
    """Get summary tensor data."""
    global SUMMARY_TENSOR_CACHE
    with _summary_lock:
        data = SUMMARY_TENSOR_CACHE
        SUMMARY_TENSOR_CACHE = {}
        return data


def _record_summary_tensor_data():
    """Record summary tensor data."""

    def check_summary_param(summary_name, tag, tensor):
        """Checks the tag is valid for summary."""
        if not isinstance(tag, str) or not tag:
            raise ValueError(f'For "{summary_name}", the name must be valid string, but got "{tag}".')
        if not isinstance(tensor, (Tensor, Tensor_)):
            raise TypeError(f'For "{summary_name}", the parameter "value" expect to be Tensor, '
                            f'but got {type(tensor).__name__}')

    summary_list = list()
    for data in debug_ops.SUMMARY_TENSOR_CACHE:
        check_summary_param(data[0], data[1], data[2])
        plugin = data[0].split('Summary')[0].lower()
        _check_to_numpy(plugin, data[2], prim=True)
        if data[0] == "TensorSummary":
            summary_op_name = data[1] + "[:Tensor]"
        elif data[0] == "ScalarSummary":
            summary_op_name = data[1] + "[:Scalar]"
        elif data[0] == "ImageSummary":
            summary_op_name = data[1] + "[:Image]"
        elif data[0] == "HistogramSummary":
            summary_op_name = data[1] + "[:Histogram]"
        summary_value = {
            "name": summary_op_name,
            "data": data[2]
        }
        summary_list.append(summary_value)
    _cache_summary_tensor_data(summary_list)
    debug_ops.SUMMARY_TENSOR_CACHE = []


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
        raise ValueError(f'For "SummaryRecord", the keys {unexpected_params} of "export_options" are unsupported, '
                         f'expect the follow keys: {list(_DEFAULT_EXPORT_OPTIONS.keys())}')

    for export_option, export_format in export_options.items():
        unexpected_format = {export_format} - _DEFAULT_EXPORT_OPTIONS.get(export_option)
        if unexpected_format:
            raise ValueError(
                f'For "SummaryRecord", the export_format {unexpected_format} of "export_options" are unsupported '
                f'for {export_option}, expect the follow values: {list(_DEFAULT_EXPORT_OPTIONS.get(export_option))}')

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
        1. When using SummaryRecord, you need to run the code in `if __name__ == "__main__"` .
        2. Make sure to close the SummaryRecord at the end, otherwise the process will not exit.
           Please see the Example section below to learn how to close properly in two ways.
        3. Only one SummaryRecord instance is allowed at a time, otherwise it will cause data writing problems.
        4. SummaryRecord only supports Linux systems.
        5. The Summary is not supported when compile source with `-s on` option.

    Args:
        log_dir (str): The log_dir is a directory location to save the summary.
        file_prefix (str): The prefix of file. Default: "events".
        file_suffix (str): The suffix of file. Default: "_MS".
        network (Cell): Obtain a pipeline through network for saving graph summary. Default: None.
        max_file_size (int, optional): The maximum size of each file that can be written to disk (in bytes).
            For example, to write not larger than 4GB, specify `max_file_size=4*1024**3`.
            Default: None, which means no limit.
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
        TypeError: `max_file_size` is not int or `file_prefix` and `file_suffix` is not string.
        ValueError: The Summary is not supported, please without `-s on` and recompile source.

    Examples:
        >>> import mindspore as ms
        >>> if __name__ == '__main__':
        ...     # use in with statement to auto close
        ...     with ms.SummaryRecord(log_dir="./summary_dir") as summary_record:
        ...         pass
        ...
        ...     # use in try .. finally .. to ensure closing
        ...     try:
        ...         summary_record = ms.SummaryRecord(log_dir="./summary_dir")
        ...     finally:
        ...         summary_record.close()
    """

    count = 0

    def __init__(self, log_dir, file_prefix="events", file_suffix="_MS",
                 network=None, max_file_size=None, raise_exception=False, export_options=None):
        self._check_count()
        with _instance_lock:
            self._check_count()
            SummaryRecord.count += 1
        if security.enable_security():
            raise ValueError('The Summary is not supported, please without `-s on` and recompile source.')
        self._event_writer = None
        self._mode, self._data_pool = 'train', defaultdict(list)
        self._status = {
            'closed': False,
            'has_graph': False
        }
        self.file_info = {
            'file_name': None,
            'file_path': None
        }
        self.base_log_dir = log_dir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.network = network
        self.max_file_size = max_file_size
        self.raise_exception = raise_exception
        self._export_options = export_options
        try:
            self._initialize()
        except (TypeError, ValueError) as err:
            SummaryRecord.count -= 1
            raise err

    def __enter__(self):
        """Enter the context manager."""
        if self._status.get('closed'):
            raise ValueError(f'For "{self.__class__.__name__}", SummaryRecord has been closed, '
                             f'please check if close() method is called')
        return self

    def __exit__(self, *err):
        """Exit the context manager."""
        self.close()

    def set_mode(self, mode):
        r"""
        Set the model running phase. Different phases affect data recording.

        Args:
            mode (str): The mode to be set, which should be 'train' or 'eval'. When the mode is 'eval',
                summary_record will not record the data of summary operators.

                - train：the model running phase is train mode.
                - eval：the model running phase is eval mode，When the mode is 'eval',
                  summary_record will not record the data of summary operators.

        Raises:
            ValueError: `mode` is not in the optional value.

        Examples:
            >>> import mindspore as ms
            >>> if __name__ == '__main__':
            ...     with ms.SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") \
            ...             as summary_record:
            ...         summary_record.set_mode('eval')
        """
        mode_spec = 'train', 'eval'
        if mode not in mode_spec:
            raise ValueError(f'For "{self.__class__.__name__}.set_mode", {repr(mode)} is not a '
                             f'recognized mode, expect the parameter "mode" is "train" or "eval"')
        self._mode = mode

    def add_value(self, plugin, name, value):
        r"""
        Add value to be recorded later.

        Args:
            plugin (str): The plugin of the value.

                - graph: the value is a computational graph.
                - scalar: the value is a scalar.
                - image: the value is an image.
                - tensor: the value is a tensor.
                - histogram: the value is a histogram.
                - train_lineage: the value is a lineage data for the training phase.
                - eval_lineage: the value is a lineage data for the evaluation phase.
                - dataset_graph: the value is a dataset graph.
                - custom_lineage_data: the value is a customized lineage data.
                - LANDSCAPE: the value is a landscape.

            name (str): The value of the name.
            value (Union[Tensor, GraphProto, TrainLineage, EvaluationLineage, DatasetGraph, UserDefinedInfo,
                LossLandscape]): The value to store.

                - The data type of value should be 'GraphProto' (see `mindspore/ccsrc/anf_ir.proto
                  <https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/utils/anf_ir.proto>`_) object
                  when the plugin is 'graph'.
                - The data type of value should be 'Tensor' object when the plugin is 'scalar', 'image', 'tensor'
                  or 'histogram'.
                - The data type of value should be a 'TrainLineage' object when the plugin is 'train_lineage',
                  see `mindspore/ccsrc/lineage.proto
                  <https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/utils/lineage.proto>`_.
                - The data type of value should be a 'EvaluationLineage' object when the plugin is 'eval_lineage',
                  see `mindspore/ccsrc/lineage.proto
                  <https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/utils/lineage.proto>`_.
                - The data type of value should be a 'DatasetGraph' object when the plugin is 'dataset_graph',
                  see `mindspore/ccsrc/lineage.proto
                  <https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/utils/lineage.proto>`_.
                - The data type of value should be a 'UserDefinedInfo' object when the plugin is 'custom_lineage_data',
                  see `mindspore/ccsrc/lineage.proto
                  <https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/utils/lineage.proto>`_.
                - The data type of value should be a 'LossLandscape' object when the plugin is 'LANDSCAPE',
                  see `mindspore/ccsrc/summary.proto
                  <https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/utils/summary.proto>`_.

        Raises:
            ValueError: `plugin` is not in the optional value.
            TypeError: `name` is not non-empty string, or the data type of value is not 'Tensor' object when the plugin
                is 'scalar', 'image', 'tensor' or 'histogram'.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> if __name__ == '__main__':
            ...     with ms.SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") \
            ...             as summary_record:
            ...         summary_record.add_value('scalar', 'loss', Tensor(0.1))
        """
        if plugin in ('tensor', 'scalar', 'image', 'histogram'):
            if not name or not isinstance(name, str):
                raise ValueError(f'For "{self.__class__.__name__}", the parameter "name" type should be str, '
                                 f'but got {type(name)}.')
            if not isinstance(value, (Tensor, Tensor_)):
                raise TypeError(f'For "{self.__class__.__name__}", the parameter "value" expect to be Tensor, '
                                f'but got {type(value).__name__}')
            np_value = _check_to_numpy(plugin, value)
            if name in {item['tag'] for item in self._data_pool[plugin]}:
                entry = repr(f'{name}/{plugin}')
                logger.warning(f'For "{self.__class__.__name__}.add_value", {entry} has duplicate values. '
                               f'Only the newest one will be recorded.')
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
        elif plugin == PluginEnum.LANDSCAPE.value:
            self._data_pool[plugin].append(dict(tag=name, value=value.SerializeToString()))
        else:
            raise ValueError(f'For "{self.__class__.__name__}.add_value", no such "plugin" of {repr(plugin)} '
                             f', expect value is one of [tensor, scalar, image, histogram, train_lineage, '
                             f'eval_lineage, dataset_graph, custom_lineage_data, graph, landscape]')

    def record(self, step, train_network=None, plugin_filter=None):
        r"""
        Record the summary.

        Args:
            step (int): Represents training step number.
            train_network (Cell): The spare network for saving graph.
                Default: None, it means just do not save the graph summary when the original network graph is None.
            plugin_filter (Callable[[str], bool], optional): The filter function, \
                which is used to filter out which plugin should be written. Default: None.

        Returns:
            bool, whether the record process is successful or not.

        Raises:
            TypeError: `step` is not int, or `train_network` is not `mindspore.nn.Cell
                <https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore-nn-cell>`_ 。

        Examples:
            >>> import mindspore as ms
            >>> if __name__ == '__main__':
            ...     with ms.SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") \
            ...             as summary_record:
            ...         result = summary_record.record(step=2)
            ...         print(result)
            ...
            True
        """
        logger.debug("SummaryRecord step is %r.", step)
        Validator.check_value_type(arg_name='step', arg_value=step, valid_types=int)
        Validator.check_value_type(arg_name='train_network', arg_value=train_network, valid_types=[Cell, type(None)])

        if self._status.get('closed'):
            logger.error(f"For '{self.__class__.__name__}', The record writer is closed, "
                         f"please check if close() method is called")
            return False
        # Set the current summary of train step
        if self.network is not None and not self._status.get('has_graph'):
            graph_proto = _cell_graph_executor.get_optimize_graph_proto(self.network)
            if graph_proto is None and train_network is not None:
                graph_proto = _cell_graph_executor.get_optimize_graph_proto(train_network)
            if graph_proto is None:
                if not context.get_context("mode") == context.PYNATIVE_MODE:
                    logger.error("Failed to get proto for graph.")
            else:
                self._event_writer.write({'graph': [{'step': step, 'value': graph_proto}]})
                self._status['has_graph'] = True
                if not SUMMARY_TENSOR_CACHE:
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

    def _initialize(self):
        """Initialize the SummaryRecord instance."""
        log_path = _make_directory(self.base_log_dir, "log_dir")

        if not isinstance(self.max_file_size, (int, type(None))):
            raise TypeError(f"For '{self.__class__.__name__}', the 'max_file_size' should be int type, "
                            f"but got type {type(self.max_file_size)}")

        if not isinstance(self.file_prefix, str) or not isinstance(self.file_suffix, str):
            raise TypeError(f"For '{self.__class__.__name__}', `file_prefix` and `file_suffix`  should be str, "
                            f"but got type {type(self.file_prefix)}")

        Validator.check_str_by_regular(self.file_prefix)
        Validator.check_str_by_regular(self.file_suffix)

        if self.max_file_size is not None and self.max_file_size < 0:
            logger.warning(f"For '{self.__class__.__name__}', the 'max_file_size' should be greater than 0. "
                           f"but got value {self.max_file_size}.")
            self.max_file_size = None

        Validator.check_value_type(arg_name='raise_exception', arg_value=self.raise_exception, valid_types=bool)

        time_second = str(int(time.time()))
        # create the summary writer file
        self.file_info['file_name'] = get_event_file_name(self.file_prefix, self.file_suffix, time_second)
        self.file_info['file_path'] = os.path.join(log_path, self.file_info.get('file_name'))

        self._export_options = process_export_options(self._export_options)
        export_dir = ''
        if self._export_options is not None:
            export_dir = "export_{}".format(time_second)

        filename_dict = dict(summary=self.file_info.get('file_name'),
                             lineage=get_event_file_name(self.file_prefix, '_lineage', time_second),
                             exporter=export_dir)
        self._event_writer = WriterPool(self.base_log_dir,
                                        self.max_file_size,
                                        self.raise_exception,
                                        **filename_dict)
        _get_summary_tensor_data()
        atexit.register(self.close)

    def _add_summary_tensor_data(self):
        """Add summary tensor data."""
        _record_summary_tensor_data()
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
        r"""
        Get the full path of the log file.

        Returns:
            str, the full path of log file.

        Examples:
            >>> import mindspore as ms
            >>> if __name__ == '__main__':
            ...     with ms.SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") \
            ...             as summary_record:
            ...         log_dir = summary_record.log_dir
        """
        return self.file_info['file_path']

    def flush(self):
        r"""
        Flush the buffer and write buffer data to disk.

        Call it to make sure that all pending events have been written to disk.

        Examples:
            >>> import mindspore as ms
            >>> if __name__ == '__main__':
            ...     with ms.SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") \
            ...             as summary_record:
            ...         summary_record.flush()
        """
        if self._status.get('closed'):
            logger.error(f"For '{self.__class__.__name__}', the record writer is closed and can not flush, "
                         f"please check if close() method is called")
        elif self._event_writer:
            self._event_writer.flush()

    def close(self):
        """
        Flush the buffer and write files to disk and close summary records. Please use the statement to autoclose.

        Examples:
            >>> import mindspore as ms
            >>> if __name__ == '__main__':
            ...     try:
            ...         summary_record = ms.SummaryRecord(log_dir="./summary_dir")
            ...     finally:
            ...         summary_record.close()
        """
        if not self._status.get('closed') and self._event_writer:
            # event writer flush and close
            logger.info('Please wait it may take quite some time to finish writing and closing.')
            atexit.unregister(self.close)
            self._event_writer.close()
            self._event_writer.join()
            self._status['closed'] = True
        with _instance_lock:
            SummaryRecord.count -= 1

    @classmethod
    def _check_count(cls):
        if cls.count > 0:
            raise RuntimeError(
                f"For '{cls.__name__}', only one instance is supported in a training process, "
                f"you are trying to create a new one when the existing instance number is {cls.count}. "
                f"Please check your scripts.")

    @staticmethod
    def _parse_from(name: str = None):
        """Parse the tag and type from name."""
        if not isinstance(name, str):
            return None, None
        match = re.match(r'(.+)\[:(.+)\]', name)
        if match:
            return match.groups()
        return None, None
