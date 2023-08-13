# Copyright 2020-201 Huawei Technologies Co., Ltd
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
from collections import defaultdict
from decimal import Decimal
from abc import abstractmethod
from enum import Enum
from pathlib import Path

from mindspore import log
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, ProfilerRawFileException
from mindspore.profiler.common.util import get_summary_for_step_trace
from mindspore.profiler.common.struct_type import StructType
from mindspore.profiler.common.util import combine_stream_task_id


class PointTag(Enum):
    """Initializing indexes."""
    MODEL_START = 0
    MODEL_END = 1
    FP_START = 2
    BP_END = 3
    ITER_END = 4
    MIN_ALL_REDUCE = 10000
    MAX_ALL_REDUCE = 20000


STEP_TRACE_RPT_TYPE = 10
TS_TRACK_STEP_TRACE_STRUCT = dict(
    mode=StructType.UINT8,
    rptType=StructType.UINT8,
    bufSize=StructType.UINT16,
    reserved1=StructType.UINT32,
    timestamp=StructType.UINT64,
    indexId=StructType.UINT64,
    modelId=StructType.UINT64,
    streamId=StructType.UINT16,
    taskId=StructType.UINT16,
    tagId=StructType.UINT16,
    reserved2=StructType.UINT16
)


class BaseStepTraceParser:
    """
    The parser for step trace data.

    Args:
        input_dir (str): The directory that contains original step trace data.
        output_file_path (str): The output file path.
        skip_first_step (bool): Whether skip the first step or not.
        is_training_mode (bool): Whether in training mode or not.
        is_gpu_kernel_async_launch (bool): Whether is gpu kernel async launch or not.
    """

    def __init__(self, input_dir, output_file_path, skip_first_step=False,
                 is_training_mode=True, is_gpu_kernel_async_launch=False):
        self._input_dir = input_dir
        self._output_path = output_file_path
        self._skip_first_step = skip_first_step
        self._is_training_mode = is_training_mode
        self._is_gpu_kernel_async_launch = is_gpu_kernel_async_launch

        self._result = []
        self._header = []
        self._step_num = 0
        self._tag_map = {}
        self._unique_id_map = {}

    @property
    def output_file(self):
        """The property of step trace header."""
        file_name = self._output_path.rsplit('/', 2)
        return file_name[-1] if len(file_name) == 3 else ''

    @staticmethod
    def _get_op_type(tag, name):
        """
        Get op type from tag and name.

        Args:
            tag (int): The tag id.
            name (str): The op name.

        Returns:
            str, the op type or communication op name.
        """
        tag_map = {PointTag.FP_START.value: 'fp', PointTag.BP_END.value: 'bp', PointTag.ITER_END.value: 'end'}
        # get solid tag type
        op_type = tag_map.get(tag, '')
        if op_type:
            return op_type
        # check if the tag is step tag.
        if tag == PointTag.MODEL_START.value:
            return 'start'
        # analyze reduce tag
        op_name = name.rsplit('/', 1)[-1]
        if not op_name:
            log.warning("Unexpected op name:%s", name)

        return op_name

    def show(self):
        """The property of step trace info."""
        summary_info = {}
        if self._result:
            summary_info = get_summary_for_step_trace(self._result[-1], self._header, self._is_training_mode)
            summary_info['total_steps'] = len(self._result) - 1
        log.info('\nStep trace summary info (unit: syscnt):')
        log.info(summary_info)
        log.info('\nThe step trace parse result saves under ${summary_dir}/profiler/%s' % self.output_file)

    def parse_and_save(self):
        """Parse step trace files and save the result."""
        self._parse()
        self._save()
        log.info("Finish to save intermediate result for step trace file.")

    @abstractmethod
    def record_point_info(self, output_path):
        """
        Record point info into json.

        Args:
            output_path (str): The output path for saving point info.

        Returns:
            dict, parsed point info.
        """

    @abstractmethod
    def _parse(self):
        """Parse source step trace files."""

    @abstractmethod
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

    def _record_trace_event(self, step_trace):
        """Record trace event."""
        log.debug("Profiler start to record trace event: %s", str(step_trace))
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
        # save the row data, The unit of duration is 10ns
        if not self._header:
            self._header = list(row_data.keys())
            log.info("Profiler step trace header: %s", str(self._header))
        self._header.extend([reduce_col for reduce_col in row_data if reduce_col not in self._header])
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

    def _record_average_info(self):
        """Calculate average info."""
        result_size = len(self._result)
        # calculate average data for each column in result data
        average_data = [0] * len(self._header)
        if result_size >= 1:
            for row_info in self._result:
                average_data = [Decimal(i) + Decimal(j) for i, j in zip(row_info, average_data)]
            average_data = [round(item / result_size) for item in average_data]
            # change step num info in average_data to None
            step_num_index = self._header.index('step_num')
            average_data[step_num_index] = '-'
        self._result.append(average_data)
        log.info("Finish add average info for step trace.")

    def _save(self):
        """save step trace file."""
        bp_point, tail, fp_duration = 5, -1, -2
        log.info("Start to save step trace file.")
        if not self._header:
            return
        try:
            with os.fdopen(os.open(self._output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                           'w') as file_handle:
                csv_writer = csv.writer(file_handle)
                if not self._is_training_mode:
                    self._header[fp_duration] = 'fp'
                    self._header = self._header[:bp_point] + self._header[bp_point + 1:tail]
                csv_writer.writerow(self._header)
                for row_data in self._result:
                    if not self._is_training_mode:
                        row_data[fp_duration] += row_data[tail]
                        row_data = row_data[:bp_point] + row_data[bp_point + 1:tail]
                    csv_writer.writerow(row_data)
            os.chmod(self._output_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            log.warning('Failed to save step trace raw info. %s', err)
            raise ProfilerIOException from err


class GpuStepTraceParser(BaseStepTraceParser):
    """The parser for gpu step trace data."""

    def __init__(self, *args, **kwargs):
        super(GpuStepTraceParser, self).__init__(*args, **kwargs)
        self._source_file_path = self._input_dir
        self._reduce_op_type = []

    def get_fp_bp(self, f_obj, all_step_fp, all_step_bp):
        """Parser the fp and bp."""
        fp_start, bp_end = 0, 1
        if self._is_gpu_kernel_async_launch:
            for line in f_obj:
                line = line.strip().split()
                all_step_fp.append(line[1].split(',')[0])
                all_step_bp.append(line[2].split(',')[0])
        else:
            lines = f_obj.readlines()
            all_step_fp.append(lines[fp_start].split()[0])
            all_step_bp.append(lines[bp_end].split()[0])

    def record_point_info(self, output_path):
        """
        Record point info into json.

        Args:
            output_path (str): The output path for saving point info.

        Returns:
            dict, parsed point info.
        """
        all_step_points = []
        all_step_fp = []
        all_step_bp = []
        try:
            with open(self._source_file_path, 'r') as f_obj:
                self.get_fp_bp(f_obj, all_step_fp, all_step_bp)
        except (IOError, OSError) as err:
            log.warning(f'Failed to read {self._source_file_path}', err)
            raise ProfilerIOException from err

        for fp_name, bp_name in zip(all_step_fp, all_step_bp):
            if self._is_training_mode:
                points = {
                    'fp_start': fp_name,
                    'bp_end': bp_name
                }
            else:
                points = {
                    'fp_start': fp_name,
                }
            all_step_points.append(points)

        try:
            with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as json_file:
                if self._is_gpu_kernel_async_launch:
                    json.dump(all_step_points, json_file)
                else:
                    json.dump(all_step_points[0], json_file)
            os.chmod(output_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            log.warning('Failed to save point info. %s', err)
            raise ProfilerIOException from err

        return all_step_points[0]

    def _parse(self):
        if self._is_gpu_kernel_async_launch:
            self._parse_async_launch()
        else:
            self._parse_not_async_launch()

    def _parse_not_async_launch(self):
        """Parse source step trace files."""
        log.info("Start to parse step trace file.")
        fp_start, bp_end, iter_end, iter_start = 0, 1, 2, 3
        reduce_start = 4
        start_time, end_time = 0, 1
        step_trace_point_count = 3

        source_file = self._source_file_path
        try:
            with open(source_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < step_trace_point_count:
                    raise ProfilerRawFileException(
                        f"Failed to parse {source_file} file. The FP_POINT/BP_POINT/ITER_END_POINT "
                        f"do not recognized correctly. Try to set the environment variable'PROFILING_FP_START' "
                        f"and 'PROFILING_BP_END' to solve this problem. For example, "
                        f"'export PROFILING_FP_START=Default/xxx/Conv2d-op1' ")
                step_trace_info_all = [line.strip().split()[1:] for line in lines]
                num_of_step = len(step_trace_info_all[0])
                for step_trace_point in step_trace_info_all:
                    if len(step_trace_point) != num_of_step:
                        raise ProfilerRawFileException(
                            f"Failed to parse {source_file} file. Due to the profiled "
                            f"step_num of FP/BP/ITER_END Point are not equal")
                iter_start_info = [step_trace_info_all[fp_start][0]] + \
                                  step_trace_info_all[iter_end][:num_of_step]
                step_trace_info_all.insert(iter_start, iter_start_info)
        except (IOError, OSError) as err:
            log.warning(f'Failed to read {source_file}', err)
            raise ProfilerIOException from err
        finally:
            pass

        for step_num in range(num_of_step):
            step_trace = {
                'start': int(step_trace_info_all[iter_start][step_num].split(',')[start_time]),
                'fp': int(step_trace_info_all[fp_start][step_num].split(',')[start_time]),
                'bp': int(step_trace_info_all[bp_end][step_num].split(',')[end_time]),
                'end': int(step_trace_info_all[iter_end][step_num].split(',')[end_time]),
                'reduce': {}
            }
            num_of_step_point = len(step_trace_info_all)
            if num_of_step_point > reduce_start:
                reduce_info = {}
                reduce_time_info = []
                for reduce_idx in range(reduce_start, num_of_step_point):
                    cur_reduce_time = step_trace_info_all[reduce_idx][step_num]
                    reduce_time_info += cur_reduce_time.split(',')
                reduce_info['ops'] = reduce_time_info
                step_trace['reduce'] = reduce_info
            self._record_trace_event(step_trace)
        self._record_average_info()
        log.info("Finish to parse step trace file.")

    def _parse_one_step(self, line):
        """
        Parse step text line to dict obj.

        Args:
            line (str): The step trace line text, it contains five parts, each part is separated by a space.
                part 1: start_op_name,start_op_time
                part 2: fp_op_name,fp_time
                part 3: bp_op_name,bp_time
                part 4: end_op_name,end_time
                part 5: [reduce_op_name,reduce1_start],it contains multiple reduce, each reduce is separated by a space.
        """

        line = line.strip().split()
        start_time = int(line[0].split(',')[1][:-1])
        fp_time = int(line[1].split(',')[1][:-1])
        bp_time = int(line[2].split(',')[1][:-1])
        end_time = int(line[3].split(',')[1][:-1])
        reduce_info = {}
        reduce_time_info = []

        for reduce_item in line[4:]:
            # add communication op start and end time, time unit from ns to 10ns.
            reduce_time_info.append(reduce_item.split(',')[1][:-1])
            reduce_time_info.append(reduce_item.split(',')[2][:-1])
            self._reduce_op_type.append(reduce_item.split(',')[0].split('/')[-1])
        step_trace = {
            'start': start_time,
            'fp': fp_time,
            'bp': bp_time,
            'end': end_time
        }
        if reduce_time_info:
            reduce_info['ops'] = reduce_time_info
        step_trace['reduce'] = reduce_info
        self._record_trace_event(step_trace)

    def _parse_async_launch(self):
        """Parse source step trace files generated from async launch kernel."""
        log.info("Start to parse step trace file.")
        try:
            with open(self._source_file_path, 'r') as f_obj:
                for line in f_obj:
                    self._parse_one_step(line)

        except (IOError, OSError) as err:
            log.warning(f'Failed to read {self._source_file_path}', err)
            raise ProfilerIOException from err

        self._record_average_info()
        log.info("Finish to parse step trace file.")

    def _get_single_reduce_event_info(self, field_name, start_point, end_point):
        """
        Get single reduce info.

        Args:
            field_name (str): The field name.
            start_point (str): Start point time.
            end_point (str): End point time.

        Returns:
            dict, reduce info.
        """
        reduce_info = {}

        index = int(field_name.split('_')[2])
        op_type = self._reduce_op_type[index]
        # append field name with op type.
        field_name += '_' + op_type
        reduce_info[field_name] = int(end_point) - int(start_point)
        reduce_info[field_name + '_start_point'] = start_point
        reduce_info[field_name + '_end_point'] = end_point

        return reduce_info


class AscendStepTraceParser(BaseStepTraceParser):
    """The parser for ascend step trace data."""

    def __init__(self, *args, **kwargs):
        super(AscendStepTraceParser, self).__init__(*args, **kwargs)
        self._task_id_op_name_dict = {}

    @staticmethod
    def _list_ts_track_files(input_dir):
        """Ts track files have 4 types data, this function will list all files."""
        step_trace_paths = []
        data_dir = os.path.join(input_dir, 'data')
        data_dir = os.path.realpath(data_dir)
        for file in Path(data_dir).glob(r'ts_track*[0-9]'):
            step_trace_paths.append(file.resolve())
        if not step_trace_paths:
            raise ProfilerRawFileException(f"Can not find any ts track files in {data_dir} when parse profiler data.")
        step_trace_paths.sort()
        log.info("Profiler found %d ts track files.", len(step_trace_paths))
        return step_trace_paths

    @staticmethod
    def _is_all_reduce_tag(tag):
        return PointTag.MIN_ALL_REDUCE.value <= tag < PointTag.MAX_ALL_REDUCE.value

    @staticmethod
    def _list_ts_track_step_traces(ts_track_paths):
        """List all ts track from ts track files."""
        step_trace_size = StructType.sizeof(TS_TRACK_STEP_TRACE_STRUCT)
        ts_tracks = []
        for path in ts_track_paths:
            try:
                with open(path, 'rb') as fp:
                    while True:
                        binary_data = fp.read(step_trace_size)
                        if len(binary_data) < step_trace_size:
                            break
                        unpacked_data = StructType.unpack_binary_data(TS_TRACK_STEP_TRACE_STRUCT, binary_data)
                        if unpacked_data.get('rptType') != STEP_TRACE_RPT_TYPE:
                            continue
                        ts_tracks.append(unpacked_data)
            except (IOError, OSError) as err:
                log.critical("Can not parse profiler file, open file %s failed, detail: %s.", path, str(err))
                raise ProfilerIOException() from err
            finally:
                pass
        log.info("Profiler found %d ts track step trace data.", len(ts_tracks))
        return ts_tracks

    def set_task_id_op_name_dict(self, task_id_op_name_dict):
        """The operator task id matches the operator name."""
        self._task_id_op_name_dict = task_id_op_name_dict

    def record_point_info(self, output_path):
        """
        Record point info into json.

        Args:
            output_path (str): The output path for saving point info.

        Returns:
            dict, parsed point info.
        """
        point_info = self._tag_map
        if self._is_training_mode:
            points = {
                'fp_start': point_info.get(PointTag.FP_START.value, ''),
                'bp_end': point_info.get(PointTag.BP_END.value, '')
            }
        else:
            points = {
                'fp_start': point_info.get(PointTag.FP_START.value, ''),
            }
        if os.path.exists(output_path):
            return points
        try:
            with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as json_file:
                json.dump(points, json_file)
            os.chmod(output_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            log.warning('Failed to save point info. %s', err)
            raise ProfilerIOException
        return points

    def _parse(self):
        """Parse source step trace files."""
        log.info("Start to parse step trace file.")
        ts_track_paths = self._list_ts_track_files(self._input_dir)
        ts_tracks = self._list_ts_track_step_traces(ts_track_paths)
        self._unique_id_map, self._tag_map = self._construct_point_info(ts_tracks, self._task_id_op_name_dict)
        self._save_step_trace_to_result(ts_tracks, self._skip_first_step)
        self._record_average_info()
        log.info("Finish to parse step trace file.")

    def _construct_point_info(self, ts_tracks, task_id_op_name_dict):
        """This function can not support multi graph scenario."""
        unique_id_map_tag = {}
        tag_map_unique_id = {}
        for ts_track in ts_tracks:
            unique_id = combine_stream_task_id(ts_track.get('streamId'), ts_track.get('taskId'))
            unique_id_map_tag[unique_id] = ts_track.get('tagId')
            tag_map_unique_id[ts_track.get('tagId')] = unique_id
        unique_id_map_op = {}
        tag_map_op = {}
        for unique_id, tag in unique_id_map_tag.items():
            unique_id_map_op[unique_id] = self._get_real_point_op_name(tag, unique_id, task_id_op_name_dict)
        for tag, unique_id in tag_map_unique_id.items():
            tag_map_op[tag] = self._get_real_point_op_name(tag, unique_id, task_id_op_name_dict)
        return unique_id_map_op, tag_map_op

    def _get_real_point_op_name(self, tag, profiling_task_id, task_id_op_name_dict):
        """Get real point op name from given tag and task id."""
        # Currently, the given task id belongs to the profiling operator. We need to obtain the operator whose
        # point is actually performed based on the tag.
        # Inserting point operator rules:
        # 1. model start profiling op -> fp start profiling op -> init-data op -> bp end profiling op -> iter end
        # 2. model start -> other op... -> fp start -> Conv op ... -> bp end -> other op -> iter end
        # 3. AllReduce profiling-op (tag:10000) -> AllReduce op -> AllReduce profiling op (tag: 10001)
        task_ids = list(task_id_op_name_dict.keys())
        op_names = list(task_id_op_name_dict.values())

        cur_task_index = task_ids.index(profiling_task_id)
        if tag == PointTag.MODEL_START.value:
            real_index = cur_task_index + 1
            is_fp_start_profiling_op = bool('Profiling-op' in op_names[real_index])
            if is_fp_start_profiling_op:
                real_index += 1
        elif tag == PointTag.FP_START.value:
            real_index = cur_task_index + 1
        elif tag in (PointTag.BP_END.value, PointTag.ITER_END.value, PointTag.MODEL_END.value):
            real_index = cur_task_index - 1
        elif tag == PointTag.ITER_END.value:
            real_index = cur_task_index - 1
        elif self._is_all_reduce_tag(tag):
            if tag % 2:
                real_index = cur_task_index - 1
            else:
                real_index = cur_task_index + 1
        else:
            real_index = cur_task_index
            log.warning("The tag id %s can not be identified.", tag)
        return op_names[real_index]

    def _save_step_trace_to_result(self, ts_tracks, skip_step):
        """Save step trace data to result."""
        step_trace = {'reduce': defaultdict(list), 'start': '-'}
        model_ids = set()
        for ts_track in ts_tracks:
            if ts_track.get('rptType') != STEP_TRACE_RPT_TYPE:
                continue
            self._construct_step_trace(ts_track, step_trace)
            model_ids.add(ts_track["modelId"])

            if step_trace.get('end'):
                if not skip_step:
                    self._record_trace_event(step_trace)
                skip_step = False
                start_time = step_trace.get('end', '-')
                step_trace.clear()
                step_trace['start'] = start_time
                step_trace['reduce'] = defaultdict(list)

        if len(model_ids) > 1:
            log.warning("[profiler] Current model has multiple sub graphs, "
                        "the segmentation of steps may be inaccurate.")

    def _construct_step_trace(self, ts_track, step_trace):
        """Construct step point data."""
        timestamp = ts_track['timestamp']
        tag_id = ts_track['tagId']
        stream_id = ts_track['streamId']

        if tag_id == PointTag.FP_START.value:
            step_trace['fp'] = timestamp
        elif tag_id == PointTag.BP_END.value:
            step_trace['bp'] = timestamp
        elif tag_id == PointTag.ITER_END.value:
            step_trace['end'] = timestamp
        elif self._is_all_reduce_tag(tag_id):
            unique_id = combine_stream_task_id(ts_track.get('streamId'), ts_track.get('taskId'))
            step_trace['reduce'][stream_id].append((unique_id, timestamp))

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
        if self._unique_id_map.get(end_point[0]) != self._unique_id_map.get(start_point[0]):
            log.warning("Unmatched reduce event <%s, %s>.", start_point, end_point)
            return reduce_info
        op_type = self._unique_id_map.get(start_point[0])
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
