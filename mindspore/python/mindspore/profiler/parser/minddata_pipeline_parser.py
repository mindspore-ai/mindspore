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
"""Thr parser for parsing minddata pipeline files."""
import csv
import json
import os
import stat
from queue import Queue

from mindspore.profiler.common.exceptions.exceptions import \
    ProfilerPathErrorException, ProfilerRawFileException, \
    ProfilerDirNotFoundException
from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path


class MinddataPipelineParser:
    """
    Thr parser for parsing minddata pipeline files.

    Args:
        source_dir (str): The minddata pipeline source dir.
        device_id (str): The device ID.
        output_path (str): The directory of the parsed file. Default: `./`.

    Raises:
        ProfilerPathErrorException: If the minddata pipeline file path or
            the output path is invalid.
    """
    _raw_pipeline_file_name = 'pipeline_profiling_{}.json'
    _parsed_pipeline_file_name = 'minddata_pipeline_raw_{}.csv'
    _col_names = [
        'op_id', 'op_type', 'num_workers', 'output_queue_size',
        'output_queue_average_size', 'output_queue_length',
        'output_queue_usage_rate', 'sample_interval', 'parent_id', 'children_id'
    ]

    def __init__(self, source_dir, device_id, output_path='./'):
        self._device_id = device_id
        self._pipeline_path = self._get_pipeline_path(source_dir)
        self._save_path = self._get_save_path(output_path)

    @property
    def save_path(self):
        """
        The property of save path.

        Returns:
            str, the save path.
        """
        return self._save_path

    def parse(self):
        """
        Parse the minddata pipeline files.

        Raises:
            ProfilerRawFileException: If fails to parse the raw file of
                minddata pipeline or the file is empty.
        """
        if not self._pipeline_path:
            return
        with open(self._pipeline_path, 'r') as file:
            try:
                pipeline_info = json.load(file)
            except (json.JSONDecodeError, TypeError) as err:
                logger.warning(err)
                raise ProfilerRawFileException(
                    'Fail to parse minddata pipeline file.'
                ) from err
        if not pipeline_info:
            logger.warning('The minddata pipeline file is empty.')
            raise ProfilerRawFileException(
                'The minddata pipeline file is empty.'
            )

        self._parse_and_save(pipeline_info)

    def _get_pipeline_path(self, source_dir):
        """
        Get the minddata pipeline file path.

        Args:
            source_dir (str): The minddata pipeline source dir.

        Returns:
            str, the minddata pipeline file path.
        """
        pipeline_path = os.path.join(
            source_dir,
            self._raw_pipeline_file_name.format(self._device_id)
        )

        try:
            pipeline_path = validate_and_normalize_path(pipeline_path)
        except RuntimeError as err:
            logger.warning('Minddata pipeline file is invalid.')
            raise ProfilerPathErrorException('Minddata pipeline file is invalid.') from err
        if not os.path.isfile(pipeline_path):
            logger.warning(
                'The minddata pipeline file <%s> not found.', pipeline_path
            )
            pipeline_path = ""

        return pipeline_path

    def _get_save_path(self, output_path):
        """
        Get the save path.

        Args:
            output_path (str): The output dir.

        Returns:
            str, the save path.
        """
        try:
            output_dir = validate_and_normalize_path(output_path)
        except RuntimeError as err:
            logger.warning('Output path is invalid.')
            raise ProfilerPathErrorException('Output path is invalid.') from err
        if not os.path.isdir(output_dir):
            logger.warning('The output dir <%s> not found.', output_dir)
            raise ProfilerDirNotFoundException(output_dir)
        return os.path.join(
            output_dir, self._parsed_pipeline_file_name.format(self._device_id)
        )

    def _parse_and_save(self, pipeline_info):
        """
        Parse and save the parsed minddata pipeline file.

        Args:
            pipeline_info (dict): The pipeline info reads from the raw file of
                the minddata pipeline.

        Raises:
            ProfilerRawFileException: If the format of minddata pipeline raw
                file is wrong.
        """
        sample_interval = pipeline_info.get('sampling_interval')
        op_info = pipeline_info.get('op_info')
        if sample_interval is None or not op_info:
            raise ProfilerRawFileException(
                'The format of minddata pipeline raw file is wrong.'
            )

        op_id_info_cache = {}
        for item in op_info:
            if not item:
                raise ProfilerRawFileException(
                    'The content of minddata pipeline raw file is wrong.'
                )
            op_id_info_cache[item.get('op_id')] = item

        with os.fdopen(os.open(self._save_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as save_file:
            csv_writer = csv.writer(save_file)
            csv_writer.writerow(self._col_names)
            self._parse_and_save_op_info(
                csv_writer, op_id_info_cache, sample_interval
            )
        os.chmod(self._save_path, stat.S_IREAD | stat.S_IWRITE)

    def _parse_and_save_op_info(self, csv_writer, op_id_info_cache,
                                sample_interval):
        """
        Parse and save the minddata pipeline operator information.

        Args:
            csv_writer (csv.writer): The csv writer.
            op_id_info_cache (dict): The operator id and information cache.
            sample_interval (int): The sample interval.

        Raises:
            ProfilerRawFileException: If the operator that id is 0 does not exist.
        """
        queue = Queue()
        root_node = op_id_info_cache.get(0)
        if not root_node:
            raise ProfilerRawFileException(
                'The format of minddata pipeline raw file is wrong, '
                'the operator that id is 0 does not exist.'
            )
        root_node['parent_id'] = None
        queue.put_nowait(root_node)

        while not queue.empty():
            node = queue.get_nowait()
            self._update_child_node(node, op_id_info_cache)
            csv_writer.writerow(self._get_op_info(node, sample_interval))

            op_id = node.get('op_id')
            children_ids = node.get('children')
            if not children_ids:
                continue
            for child_op_id in children_ids:
                sub_node = op_id_info_cache.get(child_op_id)
                sub_node['parent_id'] = op_id
                queue.put_nowait(sub_node)

    def _update_child_node(self, node, op_id_info_cache):
        """
        Updates the child node information of the operator.

        Args:
            node (dict): The node represents an operator.
            op_id_info_cache (dict): The operator id and information cache.
        """
        child_op_ids = node.get('children')
        if not child_op_ids:
            return

        queue = Queue()
        self._cp_list_item_to_queue(child_op_ids, queue)

        new_child_op_ids = []
        while not queue.empty():
            child_op_id = queue.get_nowait()
            child_node = op_id_info_cache.get(child_op_id)
            if child_node is None:
                continue
            metrics = child_node.get('metrics')
            if not metrics or not metrics.get('output_queue'):
                op_ids = child_node.get('children')
                if op_ids:
                    self._cp_list_item_to_queue(op_ids, queue)
            else:
                new_child_op_ids.append(child_op_id)

        node['children'] = new_child_op_ids

    def _get_op_info(self, op_node, sample_interval):
        """
        Get the operator information.

        Args:
            op_node (dict): The node represents an operator.
            sample_interval (int): The sample interval.

        Returns:
            list[str, int, float], the operator information.
        """
        queue_size = None
        queue_average_size = None
        queue_length = None
        queue_usage_rate = None
        metrics = op_node.get('metrics')
        if metrics:
            output_queue = metrics.get('output_queue')
            if output_queue:
                queue_size = output_queue.get('size')
                if queue_size is None:
                    raise ValueError("The queue can not be None.")
                if queue_size:
                    queue_average_size = sum(queue_size) / len(queue_size)
                queue_length = output_queue.get('length')
                if queue_length == 0:
                    raise ValueError("The length of queue can not be 0.")
                if queue_average_size is not None:
                    queue_usage_rate = queue_average_size / queue_length

        children_id = op_node.get('children')
        op_info = [
            op_node.get('op_id'),
            op_node.get('op_type'),
            op_node.get('num_workers'),
            queue_size,
            queue_average_size,
            queue_length,
            queue_usage_rate,
            sample_interval,
            op_node.get('parent_id'),
            children_id if children_id else None
        ]
        return op_info

    def _cp_list_item_to_queue(self, inner_list, queue):
        """
        Copy the contents of a list to a queue.

        Args:
            inner_list (list): The list.
            queue (Queue): The target queue.
        """
        for item in inner_list:
            queue.put_nowait(item)
