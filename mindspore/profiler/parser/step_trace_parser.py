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
"""The parser for step trace data."""
import csv
import json
import os
import stat
import struct
from collections import namedtuple
from decimal import Decimal

from mindspore.profiler.common.exceptions.exceptions import ProfilerPathErrorException, \
    JobIdMismatchException, ProfilerIOException
from mindspore import log
from mindspore.profiler.common.util import get_summary_for_step_trace

StepTraceStruct = namedtuple(
    'TrainingTraceStruct', ['tag_id', 'task_id', 'stream_id', 'sys_count']
)


class StepTraceParser:
    """
    The parser for step trace data.

    Args:
        input_dir (str): The directory that contains original step trace data.
        output_file_path (str): The output file path.
        job_id (int): The job id used to define the start of new step. Default: 0.
        skip_first_step (bool): Whether skip the first step or not.
    """
    _event_size = 20
    _fp_tag = 1
    _bp_tag = 2
    _end_tag = 255

    def __init__(self, input_dir, output_file_path, job_id=0, skip_first_step=False):
        self._input_dir = input_dir
        self._output_path = output_file_path
        self._job_id = job_id
        self._skip_first_step = skip_first_step
        self._result = []
        self._header = []
        self._step_num = 0
        self._tag_map = {}

    @property
    def output_file(self):
        """The property of step trace header."""
        file_name = self._output_path.rsplit('/', 2)
        return file_name[-1] if len(file_name) == 3 else ''

    def show(self):
        """The property of step trace info."""
        summary_info = {}
        if self._result:
            summary_info = get_summary_for_step_trace(self._result[-1], self._header)
            summary_info['total_steps'] = len(self._result) - 1
        print('\nStep trace summary info (unit: syscnt):')
        print(summary_info)
        print('\nThe step trace parse result saves under ${summary_dir}/profiler/%s'
              % self.output_file)

    def parse_and_save(self):
        """Parse step trace files and save the result."""
        try:
            source_files = self._get_step_trace_files()
            self._parse(source_files)
            self._save()
        except IOError as err:
            log.warning(err)
            raise ProfilerIOException()
        else:
            log.info("Finish to save intermediate result for step trace file.")

    def record_point_info(self, point_info, output_path):
        """
        Record point info into json.

        Args:
            point_info (dict): The point info about tag id and relative op name.
            output_path (str): The output path for saving point info.

        Returns:
            dict, parsed point info.
        """
        points = {
            'fp_start': point_info.get(self._fp_tag, ''),
            'bp_end': point_info.get(self._bp_tag, '')
        }
        try:
            with open(output_path, 'w') as json_file:
                json.dump(points, json_file)
            os.chmod(output_path, stat.S_IREAD)
        except (IOError, OSError) as err:
            log.warning('Failed to save point info. %s', err)
            raise ProfilerIOException
        return points

    def update_tag_op_type_map(self, point_info):
        """
        update the map from tag id to op type.

        Args:
            point_info (dict): The point info about tag id and relative op name.
        """
        tag_map = {}
        for tag, op_name in point_info.items():
            op_type = self._get_op_type(tag, op_name)
            tag_map[tag] = op_type
        log.info("Get tag types for step trace analysis: %s", tag_map)
        self._tag_map = tag_map

    def _get_op_type(self, tag, name):
        """
        Get op type from tag and name.

        Args:
            tag (int): The tag id.
            name (str): The op name.

        Returns:
            str, the op type.
        """
        tag_map = {self._fp_tag: 'fp', self._bp_tag: 'bp', self._end_tag: 'end'}
        # get solid tag type
        op_type = tag_map.get(tag, '')
        if op_type:
            return op_type
        # check if the tag is step tag.
        if tag > self._end_tag or tag == 0:
            return 'start'
        # analyze the reduce tag
        op_type = name.rsplit('/', 1)[-1].split('-')[0]
        if not op_type:
            log.warning("Unexpected op name:%s", name)

        return op_type

    def _get_step_trace_files(self):
        """Get step trace files."""
        # step trace files may under $profiler_dir or $profiler_dir/data
        profiler_dir = self._input_dir
        step_trace_files = self._search_file(profiler_dir)
        if not step_trace_files:
            # try to find step trace files under $profiler_dir/data
            profiler_dir = os.path.join(profiler_dir, 'data')
            step_trace_files = self._search_file(profiler_dir)
        if not step_trace_files:
            raise ProfilerPathErrorException('Training trace file does not exist.')

        return step_trace_files

    @staticmethod
    def _search_file(input_dir):
        """Search step trace file under specific input directory."""
        # validate input_dir
        if not os.path.isdir(input_dir):
            raise ProfilerPathErrorException(
                '{} does not exist or is not a dir'.format(input_dir)
            )
        # get step trace files
        files = os.listdir(input_dir)
        step_trace_files = list(
            filter(
                lambda file: file.startswith('training_trace') and not file.endswith('.done'),
                files
            )
        )
        # validate result
        if len(step_trace_files) > 1:
            # the format of file name is like
            # `training_trace.46.dev.profiler_default_tag.$id.slice_$number`
            # use the $number as the sorted key
            try:
                step_trace_files.sort(key=lambda path: int(path.rsplit('_', 1)[-1]))
            except ValueError as err:
                log.warning("Unable to parse file names: %s. %s", step_trace_files, err)
                step_trace_files = []

        file_paths = [os.path.join(input_dir, file) for file in step_trace_files]
        log.info("Find %d step trace files.", len(file_paths))
        return file_paths

    def _parse(self, source_files):
        """Parse source step trace files."""
        log.info("Start to parse step trace file.")
        event_info = {}
        for source_file in source_files:
            with open(source_file, 'rb') as handler:
                content = handler.read()
                for step_trace in self._get_next_step_trace(content, event_info):
                    if self._skip_first_step:
                        self._skip_first_step = False
                        continue
                    self._record_trace_event(step_trace)
        self._record_average_info()
        log.info("Finish to parse step trace file.")

    def _get_next_step_trace(self, content, event_info):
        """
        Get next step trace info.

        Args:
            content (bytes): The input step trace info.
            event_info (dict): The event info.

        Returns:
            Generator, return the step trace one by one.
        """
        for pos in range(0, len(content), 20):
            next_event = self._get_trace_struct(content[pos:pos + self._event_size])
            self._construct_event_info(next_event, event_info)
            if event_info.get('end'):
                yield event_info

    def _get_trace_struct(self, bin_info):
        """Translate event info to StepTraceStruct."""
        if len(bin_info) == self._event_size:
            parsed_info = struct.unpack('=QHHQ', bin_info)
            return StepTraceStruct(*parsed_info)
        return None

    def _construct_event_info(self, next_event, event_info):
        """Construct event info according to next_event."""
        min_job_id = 255
        step_flag: bool = lambda tag: tag > min_job_id or tag == 0
        end_flag: bool = lambda tag: tag == min_job_id
        fp_flag: bool = lambda tag: tag == self._fp_tag
        bp_flag: bool = lambda tag: tag == self._bp_tag

        def _on_step_event():
            """Handle step event."""
            self._validate_tag_id(tag_id)
            start_time = event_info.get('end', '-')
            event_info.clear()
            event_info['start'] = start_time
            event_info['reduce'] = {}

        def _on_reduce_event(reduce_tag_id):
            """Handle reduce event."""
            stream_id = next_event.stream_id
            if event_info['reduce'].get(stream_id):
                event_info['reduce'][stream_id].append((reduce_tag_id, sys_count))
            else:
                event_info['reduce'][stream_id] = [(reduce_tag_id, sys_count)]

        tag_id = next_event.tag_id
        sys_count = next_event.sys_count
        if end_flag(tag_id):
            event_info['end'] = sys_count
        elif step_flag(tag_id):
            _on_step_event()
        elif fp_flag(tag_id):
            event_info['fp'] = sys_count
        elif bp_flag(tag_id):
            event_info['bp'] = sys_count
        else:
            _on_reduce_event(tag_id)

    def _validate_tag_id(self, job_id):
        """Check the job id in source step trace file is same as user set."""
        if not self._job_id:
            self._job_id = job_id
        elif self._job_id != job_id:
            raise JobIdMismatchException()

    def _record_trace_event(self, step_trace):
        """Record trace event."""
        self._step_num += 1
        start_time = step_trace.get('start')
        end_time = step_trace.get('end')
        fp_time = step_trace.get('fp')
        bp_time = step_trace.get('bp')
        if not (start_time and end_time and fp_time and bp_time):
            log.warning("The step %d lacks basic time.", self._step_num)
            return
        if start_time == '-':
            start_time = fp_time
        row_data = {
            'step_num': self._step_num,
            'start_point': start_time,
            'end_point': end_time,
            'total': end_time - start_time,
            'fp_point': fp_time,
            'bp_point': bp_time,
            'iteration_interval': fp_time - start_time,
            'fp_and_bp': bp_time - fp_time,
            'tail': end_time - bp_time
        }
        # update reduce info
        self._update_reduce_info(step_trace, row_data)
        # save the row data
        if not self._header:
            self._header = list(row_data.keys())
        row_data_list = [row_data.get(header_name, 0) for header_name in self._header]
        self._result.append(row_data_list)

    def _update_reduce_info(self, step_trace, row_data):
        """Extract reduce info."""
        reduce_time = step_trace.get('reduce', {})
        for stream_id, time_points in reduce_time.items():
            time_point_num = len(time_points)
            if time_point_num % 2:
                log.warning("Stream %d has %d reduce time points.", stream_id, time_point_num)
                continue
            for index, point_id in enumerate(range(0, time_point_num, 2)):
                field_name = f'stream_{stream_id}_{index}'
                reduce_info = self._get_single_reduce_event_info(
                    field_name, time_points[point_id], time_points[point_id + 1])
                row_data.update(reduce_info)

    def _get_single_reduce_event_info(self, field_name, start_point, end_point):
        """
        Get single reduce info.

        Args:
            field_name (str): The field name.
            start_point (Tuple[int, int]): Start point time info, including (tag_id, sys_count).
            end_point (Tuple[int, int]): End point time info, including (tag_id, sys_count).

        Returns:
            dict, reduce info.
        """
        reduce_info = {}
        if end_point[0] - start_point[0] != 1 or end_point[0] % 2:
            log.warning("Unmatched reduce event <%s, %s>.", start_point, end_point)
            return reduce_info
        op_type = self._tag_map.get(start_point[0])
        # append field name with op type.
        if not op_type:
            log.warning("Can't recognize the inner type for point tag: %d.", start_point[0])
            field_name += '_parallel'
        else:
            field_name += '_' + op_type
        reduce_info[field_name] = end_point[1] - start_point[1]
        reduce_info[field_name + '_start_point'] = start_point[1]
        reduce_info[field_name + '_end_point'] = end_point[1]

        return reduce_info

    def _record_average_info(self):
        """Calculate average info."""
        result_size = len(self._result)
        # calculate average data for each column in result data
        average_data = [0] * len(self._header)
        if result_size >= 2:
            for row_info in self._result[1:]:
                average_data = [
                    Decimal(i) + Decimal(j) for i, j in zip(row_info, average_data)
                ]
            average_data = [
                round((item / (result_size - 1))) for item in average_data
            ]
            # change step num info in average_data to None
            step_num_index = self._header.index('step_num')
            average_data[step_num_index] = '-'
        self._result.append(average_data)
        log.info("Finish add average info for step trace.")

    def _save(self):
        log.info("Start to save step trace file.")
        if not self._header:
            return
        with open(self._output_path, 'w') as file_handle:
            csv_writer = csv.writer(file_handle)
            csv_writer.writerow(self._header)
            for row_data in self._result:
                csv_writer.writerow(row_data)
        os.chmod(self._output_path, stat.S_IREAD)
