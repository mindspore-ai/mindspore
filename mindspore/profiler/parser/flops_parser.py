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
"""Memory Usage Parser."""
import os
import struct
import json
import stat

from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, \
    ProfilerFileNotFoundException, ProfilerRawFileException


class FlopsParser:
    """
    The parser for parsing aicore file.

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
    _task_id_threshold = 25000

    def __init__(self, input_dir, output_dir, op_task_dict, device_id):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._op_task_dict = op_task_dict
        self._device_id = device_id
        self._flops_filename = f'flops_{self._device_id}.txt'
        self._flops_summary_filename = f'flops_summary_{self._device_id}.json'
        self._aicore_filename = f'aicore.data.{self._device_id}.slice_0'
        self._optime_filename = f'output_op_compute_time_{self._device_id}.txt'
        self._info_json = f'info.json.{device_id}'
        self._flops_summary = {
            'FLOPs': 0,
            'FLOPS': 0,
            'FLOPS_Utilization': 0
        }

    def execute(self):
        """Get the flops of aicore operators and write to file."""
        peak_flops = self._get_peak_flops()
        read_count, all_log_struct = self._load_aicore_data()
        op_avg_time_dict = self._get_op_avg_time_dict()
        op_flops_list = []
        op_name_set = set()

        for idx in range(read_count):
            log_struct = all_log_struct[idx * self.AICORE_LOG_SIZE:
                                        (idx + 1) * self.AICORE_LOG_SIZE]
            result = [hex(i) for i in struct.unpack(self.RUNTIME_COMMON, log_struct)]
            op_name = self._get_op_name(result)
            if op_name in op_name_set or op_name == "":
                continue
            # Convert the unit of task_fops to MFLOPs(1e6).
            task_fops = self._compute_task_flops(result) * 1e-6
            op_avg_time = op_avg_time_dict[op_name]
            # Time unit of op_avg_time is ms.
            # The unit of gflop_per_second is GFLOPS(1e9).
            gflop_per_second = task_fops / float(op_avg_time)
            flops_utilization = (gflop_per_second * 1e9 / peak_flops) * 100
            self._flops_summary['FLOPs'] += task_fops
            self._flops_summary['FLOPS'] += gflop_per_second
            self._flops_summary['FLOPS_Utilization'] += flops_utilization
            op_flops = [op_name, str(task_fops), str(gflop_per_second), str(flops_utilization)]
            op_flops_list.append(op_flops)
            op_name_set.add(op_name)

        self._flops_summary['FLOPS'] /= len(op_name_set)
        self._flops_summary['FLOPS_Utilization'] /= len(op_name_set)

        self._write_file(op_flops_list)

    def _load_aicore_data(self):
        """Load the original binary aicore data."""
        aicore_file_path = os.path.join(
            self._input_dir,
            "data",
            self._aicore_filename
        )

        if not os.path.exists(aicore_file_path):
            logger.error(f'The file {aicore_file_path} does not exist.')
            raise ProfilerFileNotFoundException('aicore.data')

        file_size = os.path.getsize(aicore_file_path)
        read_count = file_size // self.AICORE_LOG_SIZE

        if not read_count:
            logger.error(f'the file {aicore_file_path} '
                         f'does not have enough content to be parsed.')
            raise ProfilerRawFileException(
                'aicore.data file does not have enough content to be parsed'
            )

        try:
            with open(aicore_file_path, "rb") as aicore_file:
                all_log_struct = aicore_file.read(self.AICORE_LOG_SIZE * read_count)
        except (IOError, OSError) as err:
            logger.error(f'Error occurred when read {aicore_file_path} file: {err}')
            raise ProfilerIOException()

        return read_count, all_log_struct

    def _get_peak_flops(self):
        """Get the peak FLOPS of current ascend device."""
        info_json_file_path = os.path.join(
            self._input_dir,
            self._info_json
        )

        if not os.path.exists(info_json_file_path):
            logger.error(f'The file {info_json_file_path} does not exist.')
            raise ProfilerFileNotFoundException(info_json_file_path)

        try:
            with open(info_json_file_path, 'r', encoding='utf-8') as info_file:
                device_info = json.load(info_file)['DeviceInfo'][0]
                device_frequency = float(device_info["aic_frequency"])
                ai_core_num = float(device_info["ai_core_num"])
                # peak_flops formula (provided by Hisi): device_frequency * num_of_aicore * 4096 * 2.
                peak_flops = device_frequency * 1e6 * ai_core_num * 4096 * 2
        except (IOError, OSError, json.JSONDecodeError) as err:
            logger.error(f'Error occurred when read {info_json_file_path} file: {err}')
            raise ProfilerIOException()

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
        cube_fops = cube_fp16_exec*(16**3)*2 + cube_int8_exec*16*16*32*2
        vec_fops = vec_fp32*32 + vec_fp16_128lane_exec*128 + \
                   vec_fp16_64lane_exec*64 + vec_int32_exec*64 + vec_misc_exec*32
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
        """get the op average execution time."""
        op_avg_time_dict = {}
        optime_file_path = os.path.join(self._output_dir, self._optime_filename)

        if not os.path.exists(optime_file_path):
            logger.error(f'The {optime_file_path} file does not exist.')
            raise ProfilerFileNotFoundException(optime_file_path)

        try:
            with open(optime_file_path, 'r') as f:
                lines = f.readlines()
                op_avg_time_lines = lines[3:]  # the first three lines are table header.
                for line in op_avg_time_lines:
                    op_name, avg_time = line.split()[:2]
                    op_avg_time_dict[op_name] = avg_time
        except (IOError, OSError) as err:
            logger.error(f'Error occurred when read {optime_file_path} file: {err}')
            raise ProfilerIOException()

        return op_avg_time_dict

    def _write_file(self, op_flops_list):
        """Write the operator's flops related information into file."""
        output_file_path = os.path.join(
            self._output_dir,
            self._flops_filename
        )
        output_summary_file_path = os.path.join(
            self._output_dir,
            self._flops_summary_filename
        )

        try:
            with open(output_file_path, 'w') as f:
                header = "op_full_name, MFLOPs(10^6), GFLOPS(10^9), FLOPS utilization(%) \n"
                f.writelines(header)
                for op_flops in op_flops_list:
                    line = ", ".join(op_flops)
                    f.writelines(line + '\n')
            os.chmod(output_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.error(f'Error occurred when writing {output_file_path} file: {err}')
            raise ProfilerIOException()

        for key in self._flops_summary:
            self._flops_summary[key] = round(self._flops_summary[key], 3)
        try:
            with open(output_summary_file_path, 'w') as json_file:
                json.dump(self._flops_summary, json_file)
            os.chmod(output_summary_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.error(f'Error occurred when write {output_summary_file_path} file: {err}')
            raise ProfilerIOException()
