# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Flops parser which parsing flops from aicore file."""
import os
import struct
import json
import stat

from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, \
    ProfilerFileNotFoundException, ProfilerRawFileException, ProfilerPathErrorException
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path


class FlopsParser:
    """
    The parser for parsing flops from aicore file.

    Args:
        input_dir (str): Directory(JOBXXX) where the original profiling data are located.
        output_dir (str): Directory(profiler-{timestamp}) where the parsed profiling files are located.
        op_task_dict (dict): The mapping relation of task_id and op_full_name.
        device_id (str): The device ID.
    """
    HEX = 16
    PMU_COUNT = 8
    AICORE_LOG_SIZE = 128  # the size of each struct is 128 Byte.
    RUNTIME_COMMON = "BBHHHII10Q8I"

    # If the task id is less than the task id threshold,
    # the combination of task id and Stream id represents one operator,
    # else the task id represents one operator.
    _task_id_threshold = 65536

    def __init__(self, input_dir, output_dir, op_task_dict, device_id, rank_id, is_training_mode_flag=True):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._op_task_dict = op_task_dict
        self._device_id = device_id
        self._rank_id = rank_id
        self.is_training_mode_flag = is_training_mode_flag
        # output files.
        self._flops_filename = f'flops_{self._rank_id}.txt'
        self._flops_summary_filename = f'flops_summary_{self._rank_id}.json'
        self._flops_scope_filename = f'flops_scope_{self._rank_id}.json'
        self._flops_utilization_step_filename = f'flops_utilization_step_{self._rank_id}.json'
        # input files.
        self._aicore_filename_pref = f'aicore.data.{self._device_id}.slice'
        self._optime_filename = f'output_op_compute_time_{self._rank_id}.txt'
        self._info_json = f'info.json.{self._device_id}'
        self._step_trace_filename = f'step_trace_raw_{self._rank_id}_detail_time.csv'
        self._timeline_data_filename = f'output_timeline_data_{self._rank_id}.txt'
        self._flops_summary = {
            'FLOPs': 0,
            'FLOPS': 0,
            'FLOPS_Utilization': 0
        }
        self._flops_each_scope = {}
        self._flops_sankey_diagram = {}
        self._max_scope_num = 0

    @staticmethod
    def _read_line(start_dot, end_dot, op_avg_time_lines, op_all_step_time, op_all_step_comp):
        """Read the bp and fp time from line."""
        for op_avg_idx in op_avg_time_lines:
            line = op_avg_idx.split(',')
            fp = float(line[start_dot]) / 100000.0
            bp = float(line[end_dot]) / 100000.0
            op_all_step_time.append([fp, bp])
            op_all_step_comp.append([0.0, bp - fp])
        return op_all_step_time, op_all_step_comp

    @staticmethod
    def _add_step_flops_time(op_name, task_fops, op_idx, step_idx, op_start_time,
                             op_all_step_time, op_all_step_comp):
        """Get the start time from the current task."""
        while((op_idx < len(op_start_time)) and (op_name != op_start_time[op_idx][0])):
            op_idx += 1
        if op_idx >= len(op_start_time):
            logger.debug(f"Op name {op_name} does not exist in timeline dict.")
            return op_idx, step_idx, op_all_step_comp

        # do not add the op FLOPS that not in fp_and_bp time.
        while((step_idx < len(op_all_step_time)) and
              (op_start_time[op_idx][1] >= op_all_step_time[step_idx][1])):
            step_idx += 1
        if step_idx >= len(op_all_step_time):
            logger.info(f"Op name {op_name} does not exist in timeline dict.")

        # add the op FLOPS that in fp_and_bp time.
        if ((step_idx < len(op_all_step_time)) and
                (op_start_time[op_idx][1] >= op_all_step_time[step_idx][0]) and
                (op_start_time[op_idx][1] <= op_all_step_time[step_idx][1])):
            op_all_step_comp[step_idx][0] += task_fops
        # next op.
        op_idx += 1
        return op_idx, step_idx, op_all_step_comp

    def execute(self):
        """Get the flops of aicore operators and write to file."""
        peak_flops = self._get_peak_flops()

        op_avg_time_dict = self._get_op_avg_time_dict()
        op_flops_list = []
        op_name_set = set()
        op_compute_dict = dict()
        # get all step time.
        op_all_step_time, op_all_step_comp = self._get_all_step_time()
        op_start_time = self._get_op_start_time()
        op_idx = 0
        step_idx = 0
        aicore_file_doc = os.path.join(self._input_dir, "data")
        source_files = self._get_aicore_files(aicore_file_doc)
        if not source_files:
            return
        # parse all sliced aicore files.
        for source_file in source_files:
            source_file = validate_and_normalize_path(source_file)
            read_count, all_log_struct = self._load_aicore_data(source_file)

            for idx in range(read_count):
                log_struct = all_log_struct[idx * self.AICORE_LOG_SIZE:
                                            (idx + 1) * self.AICORE_LOG_SIZE]
                result = [hex(i) for i in struct.unpack(self.RUNTIME_COMMON, log_struct)]
                op_name = self._get_op_name(result)

                if op_name == "":
                    # filter out the blank line in the file.
                    continue
                if op_name not in op_avg_time_dict:
                    logger.info(f"Op name {op_name} does not exist in op average time dict.")
                    continue
                # Convert the unit of task_fops to MFLOPs(1e6).
                if op_name in op_compute_dict:
                    task_fops = op_compute_dict.get(op_name)
                else:
                    task_fops = self._compute_task_flops(result) * 1e-6
                    op_compute_dict[op_name] = task_fops

                # add the op FLOPS in current step.
                if len(op_start_time) >= 1 and len(op_all_step_time) >= 1:
                    op_idx, step_idx, op_all_step_comp = self._add_step_flops_time(
                        op_name, task_fops, op_idx, step_idx, op_start_time, op_all_step_time, op_all_step_comp)

                # calculate averge op FLOPS.
                if op_name in op_name_set:
                    continue
                op_avg_time = op_avg_time_dict.get(op_name)
                # Time unit of op_avg_time is ms.
                # The unit of gflop_per_second is GFLOPS(1e9).
                if float(op_avg_time) == 0.0:
                    raise ValueError("All operators take 0 ms.")
                if peak_flops == 0:
                    raise ValueError("The frequency of an operator is 0.")
                gflop_per_second = task_fops / float(op_avg_time)
                flops_utilization = (gflop_per_second * 1e9 / peak_flops) * 100
                self._flops_summary['FLOPs'] += task_fops
                self._flops_summary['FLOPS'] += gflop_per_second
                op_flops = [op_name, str(task_fops), str(gflop_per_second), str(flops_utilization)]
                op_flops_list.append(op_flops)
                op_name_set.add(op_name)
                self._add_flops_to_each_scope(op_name, task_fops)

        if not op_name_set:
            raise ProfilerRawFileException("No aicore operator found.")
        self._flops_summary['FLOPS'] /= len(op_name_set)

        sum_flops_utilization = 0.0
        # calculate the every step FLOPS utilization and the average values.
        utilization_save_filename = os.path.join(self._output_dir, self._flops_utilization_step_filename)
        with os.fdopen(os.open(utilization_save_filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
            f.write("steps, FLOPS_Utilization %\n")
            for i, x in enumerate(op_all_step_comp):
                current_utilization = x[0] / x[1] * 1e9 / peak_flops * 100
                sum_flops_utilization += current_utilization
                f.write(str(i + 1))
                f.write(",")
                f.write(str(current_utilization))
                f.write("\n")
        os.chmod(utilization_save_filename, stat.S_IREAD | stat.S_IWRITE)

        if len(op_all_step_comp) >= 1:
            self._flops_summary['FLOPS_Utilization'] = sum_flops_utilization / len(op_all_step_comp)
        else:
            logger.warning("The number of data calculation steps is 0, please check whether the "
                           "output timeline data is none.")
            self._flops_summary['FLOPS_Utilization'] = 0.0
        self._format_scope_flops()
        self._write_file(op_flops_list)

    def _load_aicore_data(self, aicore_file_path):
        """Load the original binary aicore data."""
        logger.info("the aicore file path is %s", aicore_file_path)

        if not os.path.exists(aicore_file_path):
            logger.critical(f'The file {aicore_file_path} does not exist.')
            raise ProfilerFileNotFoundException('aicore.data')

        file_size = os.path.getsize(aicore_file_path)
        read_count = file_size // self.AICORE_LOG_SIZE

        if not read_count:
            logger.critical(f'the file {aicore_file_path} '
                            f'does not have enough content to be parsed.')
            raise ProfilerRawFileException(
                'aicore.data file does not have enough content to be parsed'
            )
        aicore_file_path = os.path.realpath(aicore_file_path)
        try:
            with open(aicore_file_path, "rb") as aicore_file:
                all_log_struct = aicore_file.read(self.AICORE_LOG_SIZE * read_count)
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {aicore_file_path} file: {err}')
            raise ProfilerIOException() from err

        return read_count, all_log_struct

    def _get_peak_flops(self):
        """Get the peak FLOPS of current ascend device."""
        info_json_file_path = os.path.join(
            self._input_dir,
            self._info_json
        )

        if not os.path.exists(info_json_file_path):
            logger.critical(f'The file {info_json_file_path} does not exist.')
            raise ProfilerFileNotFoundException(info_json_file_path)

        try:
            with open(info_json_file_path, 'r', encoding='utf-8') as info_file:
                device_info = json.load(info_file)['DeviceInfo'][0]
                device_frequency = float(device_info["aic_frequency"])
                ai_core_num = float(device_info["ai_core_num"])
                # peak_flops formula (provided by Hisi): device_frequency * num_of_aicore * 4096 * 2.
                peak_flops = device_frequency * 1e6 * ai_core_num * 4096 * 2
        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.critical(f'Error occurred when read {info_json_file_path} file: {err}')
            raise ProfilerIOException() from err

        return peak_flops

    def _compute_task_flops(self, log_result):
        """Compute the FLOPs of each task."""
        pmu_cnt = list(
            int(i.replace('0x', ''), self.HEX)
            for i in log_result[9:17]
        )
        cube_fp16_exec = pmu_cnt[0]
        cube_int8_exec = pmu_cnt[1]
        vec_fp32 = pmu_cnt[2]
        vec_fp16_128lane_exec = pmu_cnt[3]
        vec_fp16_64lane_exec = pmu_cnt[4]
        vec_int32_exec = pmu_cnt[5]
        vec_misc_exec = pmu_cnt[6]

        # These formula is provided by HISI profiling.
        # a cube_fp16 instruction has (16**3)*2 float point operation.
        # a cube_fp16 instruction has 16*16*32*2 float point operation.
        cube_fops = cube_fp16_exec * (16 ** 3) * 2 + cube_int8_exec * 16 * 16 * 32 * 2
        vec_fops = vec_fp32 * 32 + vec_fp16_128lane_exec * 128 + \
                   vec_fp16_64lane_exec * 64 + vec_int32_exec * 64 + vec_misc_exec * 32
        task_fops = cube_fops + vec_fops

        return task_fops

    def _get_op_name(self, log_result):
        """Get the operator name for current task_id."""
        task_id = int(log_result[4].replace('0x', ''), self.HEX)
        stream_id = int(log_result[17].replace('0x', ''), self.HEX)
        if task_id < self._task_id_threshold:
            task_id = '_'.join([str(stream_id), str(task_id)])
        if str(task_id) not in self._op_task_dict:
            return ""
        op_name = self._op_task_dict[str(task_id)]

        return op_name

    def _get_op_avg_time_dict(self):
        """Get the op average execution time."""
        op_avg_time_dict = {}
        optime_file_path = os.path.join(self._output_dir, self._optime_filename)

        if not os.path.exists(optime_file_path):
            logger.critical(f'The {optime_file_path} file does not exist.')
            raise ProfilerFileNotFoundException(optime_file_path)
        optime_file_path = os.path.realpath(optime_file_path)
        try:
            with open(optime_file_path, 'r') as f:
                lines = f.readlines()
                op_avg_time_lines = lines[3:]  # the first three lines are table header.
                for line in op_avg_time_lines:
                    op_name, avg_time = line.split()[:2]
                    op_avg_time_dict[op_name] = avg_time
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {optime_file_path} file: {err}')
            raise ProfilerIOException() from err

        return op_avg_time_dict

    def _add_flops_to_each_scope(self, op_name, task_fops):
        """
        Add task_fops to each scope of the op_name.

        The top-level scope name is "Default" or "recompute_Default" or "Gradients".
        To classify the same scope name under different top-level scope, a suffix name is added.
        For op_name like "Default/network", the "network" will be renamed as "network(Default)".
        For op_name like "recompute_Default/network", "network" --> "network(recompute_Default)".
        For op_name like "Gradients/network", "network" --> "network(Gradients)".
        For op_name like "Gradients/recompute_Default/network", "network" --> "network(recompute_Gradients)".
        """
        # Only extracts the scope name, remove the operator name.
        scope_list = op_name.split('/')[:-1]
        self._max_scope_num = max(self._max_scope_num, len(scope_list))
        top_level_scope = scope_list[0]
        suffix_name = ""
        if op_name.startswith("Gradients/recompute_Default"):
            suffix_name = "recompute_Gradients"
        else:
            suffix_name = top_level_scope
        # To distinguish the same scope name at different scope level and different top level scope,
        # the scope level and top level scope is added.
        for level, scope_name in enumerate(scope_list):
            scope_list[level] = scope_name + f"({level}_{suffix_name})"
        scope_list[0] = top_level_scope

        # Add root node (refers to total flops).
        scope_list.insert(0, "Total")
        scope_depth = len(scope_list)
        for idx in range(scope_depth - 1):
            key_name = '{} {}'.format(scope_list[idx], scope_list[idx + 1])
            self._flops_each_scope.setdefault(key_name, 0)
            self._flops_each_scope[key_name] += task_fops

    def _format_scope_flops(self):
        """
        Format the flops of each scope to a Sankey Diagram.

        The format of Sankey Diagram is:
            {"nodes": [
                        {"name": "Default"},
                        {"name": "network"}
                     ],
             "links": [
                        {"source": "Total", "target": "Default", "value": 555},
                        {"source": "Default", "target": "network", "value": 555}
                     ]
            }
        """
        nodes, links = [], []
        scope_name_set = set()
        for scope_link, task_fops in self._flops_each_scope.items():
            source, target = scope_link.split()
            scope_name_set.update({source, target})
            link = {
                "source": source,
                "target": target,
                "value": round(task_fops, 3)
            }
            links.append(link)

        for scope_name in scope_name_set:
            node = {"name": scope_name}
            nodes.append(node)

        sankey_diagram = {"nodes": nodes, "links": links}
        self._flops_sankey_diagram = {
            "data": sankey_diagram,
            "max_scope_num": self._max_scope_num
        }

    def _write_file(self, op_flops_list):
        """Write the operator's flops related information into file."""
        join_file_path = lambda x: os.path.join(self._output_dir, x)
        output_file_path = join_file_path(self._flops_filename)
        output_summary_file_path = join_file_path(self._flops_summary_filename)
        output_flops_scope_file_path = join_file_path(self._flops_scope_filename)

        try:
            with os.fdopen(os.open(output_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
                header = "op_full_name, MFLOPs(10^6), GFLOPS(10^9), FLOPS utilization(%) \n"
                f.writelines(header)
                for op_flops in op_flops_list:
                    line = ", ".join(op_flops)
                    f.writelines(line + '\n')
            os.chmod(output_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when writing {output_file_path} file: {err}')
            raise ProfilerIOException() from err

        for key in self._flops_summary:
            self._flops_summary[key] = round(self._flops_summary[key], 3)
        try:
            with os.fdopen(os.open(output_summary_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                           'w') as json_file:
                json.dump(self._flops_summary, json_file)
            os.chmod(output_summary_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when write {output_summary_file_path} file: {err}')
            raise ProfilerIOException() from err

        try:
            with os.fdopen(os.open(output_flops_scope_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                           'w') as json_file:
                json.dump(self._flops_sankey_diagram, json_file)
            os.chmod(output_flops_scope_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when write {output_flops_scope_file_path} file: {err}')
            raise ProfilerIOException() from err

    def _get_aicore_files(self, profiler_dir):
        """Get aicore files."""
        aicore_files = self._search_file(profiler_dir)
        if not aicore_files:
            logger.warning("Aicore file does not exist.")
            return[]

        return aicore_files

    def _search_file(self, input_dir):
        """Search aicore file under specific input directory."""
        # validate input_dir
        if not os.path.isdir(input_dir):
            raise ProfilerPathErrorException(
                '{} does not exist or is not a dir'.format(input_dir)
            )
        # get aicore files
        files = os.listdir(input_dir)
        aicore_files = list(
            filter(
                lambda file: file.startswith(self._aicore_filename_pref) and not file.endswith('.done'),
                files
            )
        )
        # validate result
        if len(aicore_files) > 1:
            # the format of file name is like `aicore.data.$id.slice_$number`.
            # use the $number as the sorted key
            try:
                aicore_files.sort(key=lambda path: int(path.rsplit('_', 1)[-1]))
            except ValueError as err:
                logger.warning("Unable to parse file names: %s. %s", aicore_files, err)
                aicore_files = []
        else:
            aicore_files = list(
                filter(
                    lambda file: file.startswith(self._aicore_filename_pref) and not file.endswith('.done'),
                    files
                )
            )
            if len(aicore_files) >= 1:
                logger.warning("The aicore file structure is changed, please upgrade " \
                    "mindspore and regenerate profiling data")

        file_paths = [os.path.join(input_dir, file) for file in aicore_files]
        logger.info("Find %d aicore files.", len(file_paths))
        return file_paths

    def _get_all_step_time(self):
        """Get the op average execution time."""
        op_all_step_time = []
        op_all_step_comp = []
        _step_trace_file_path = os.path.join(self._output_dir, self._step_trace_filename)

        if not os.path.exists(_step_trace_file_path):
            logger.warning(f'The {_step_trace_file_path} file does not exist.')
            return op_all_step_time, op_all_step_comp
        try:
            with open(_step_trace_file_path, 'r') as f:
                lines = f.readlines()
                op_all_step_time, op_all_step_comp = \
                    self._get_bp_fp_time_by_line(lines, op_all_step_time, op_all_step_comp)
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {_step_trace_file_path} file: {err}')
            raise ProfilerIOException() from err
        logger.info("the train step is %d .", len(op_all_step_time))
        if not op_all_step_time:
            logger.warning(f'Empty when read {_step_trace_file_path} file, please check the valid'
                           'data of this file.')
        return op_all_step_time, op_all_step_comp

    def _get_bp_fp_time_by_line(self, lines, op_all_step_time, op_all_step_comp):
        """Get the bp and fp time with lines."""
        # the last line is the average info.
        op_avg_time_lines = lines[1:-1]
        # train mode.
        if self.is_training_mode_flag:
            op_all_step_time, op_all_step_comp = \
                self._read_line(4, 5, op_avg_time_lines, op_all_step_time, op_all_step_comp)
        else:
            # eval mode.
            op_all_step_time, op_all_step_comp = \
                self._read_line(4, 2, op_avg_time_lines, op_all_step_time, op_all_step_comp)
        return op_all_step_time, op_all_step_comp

    def _get_op_start_time(self):
        """Get the op average execution time."""
        op_start_time = []
        _timeline_file_path = os.path.join(self._output_dir, self._timeline_data_filename)

        if not os.path.exists(_timeline_file_path):
            logger.critical(f'The {_timeline_file_path} file does not exist.')
            raise ProfilerFileNotFoundException(_timeline_file_path)
        try:
            with open(_timeline_file_path, 'r') as f:
                lines = f.readlines()
                op_avg_time_lines = lines[1:]
                for op_avg_idx in op_avg_time_lines:
                    line = op_avg_idx.split(',')
                    op_name = line[0]
                    op_start = float(line[2])
                    op_start_time.append([op_name, op_start])
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {_timeline_file_path} file: {err}')
            raise ProfilerIOException() from err
        if not op_start_time:
            logger.warning(f'Empty when read {_timeline_file_path} file, please check the valid'
                           'data of this file.')
        return op_start_time
