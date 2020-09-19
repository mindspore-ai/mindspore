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
"""The parser for hwts log file."""
import os
import struct
from mindspore.profiler.common.util import fwrite_format, get_file_join_name
from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path

class HWTSLogParser:
    """
    The Parser for hwts log files.

    Args:
         input_path (str): The profiling job path. Such as: '/var/log/npu/profiling/JOBAIFGJEJFEDCBAEADIFJAAAAAAAAAA".
         output_filename (str): The output data path and name. Such as: './output_format_data_hwts_0.txt'.
    """

    _source_file_target_old = 'hwts.log.data.45.dev.profiler_default_tag'
    _source_file_target = 'hwts.data'
    _dst_file_title = 'title:45 HWTS data'
    _dst_file_column_title = 'Type           cnt  Core_ID  Block_ID  Task_ID  Cycle_counter   Stream_ID'

    def __init__(self, input_path, output_filename):
        self._input_path = input_path
        self._output_filename = output_filename
        self._source_flie_name = self._get_source_file()

    def _get_source_file(self):
        """Get hwts log file name, which was created by ada service."""

        file_name = get_file_join_name(self._input_path, self._source_file_target)
        if not file_name:
            file_name = get_file_join_name(self._input_path, self._source_file_target_old)
            if not file_name:
                data_path = os.path.join(self._input_path, "data")
                file_name = get_file_join_name(data_path, self._source_file_target)
                if not file_name:
                    file_name = get_file_join_name(data_path, self._source_file_target_old)
                    if not file_name:
                        msg = "Fail to find hwts log file, under profiling directory"
                        raise RuntimeError(msg)

        return file_name

    def execute(self):
        """
        Execute the parser, get result data, and write it to the output file.

        Returns:
            bool, whether succeed to analyse hwts log.
        """

        content_format = ['QIIIIIIIIIIII', 'QIIQIIIIIIII', 'IIIIQIIIIIIII']
        log_type = ['Start of task', 'End of task', 'Start of block', 'End of block', 'Block PMU']

        result_data = ""

        self._source_flie_name = validate_and_normalize_path(self._source_flie_name)
        with open(self._source_flie_name, 'rb') as hwts_data:
            while True:
                # read 64 bit data
                line = hwts_data.read(64)
                if line:
                    if not line.strip():
                        continue
                else:
                    break
                byte_first_four = struct.unpack('BBHHH', line[0:8])
                # byte_first[0:4] refers to count. byte_first[4] refers to is_warn_res0_0v.
                # byte_first[5:8] refers to the type of ms.
                byte_first = bin(byte_first_four[0]).replace('0b', '').zfill(8)
                ms_type = byte_first[-3:]
                is_warn_res0_ov = byte_first[4]
                cnt = int(byte_first[0:4], 2)
                core_id = byte_first_four[1]
                blk_id, task_id = byte_first_four[3], byte_first_four[4]
                if ms_type in ['000', '001', '010']:  # log type 0,1,2
                    result = struct.unpack(content_format[0], line[8:])
                    syscnt = result[0]
                    stream_id = result[1]
                elif ms_type == '011':  # log type 3
                    result = struct.unpack(content_format[1], line[8:])
                    syscnt = result[0]
                    stream_id = result[1]
                elif ms_type == '100':  # log type 4
                    result = struct.unpack(content_format[2], line[8:])
                    stream_id = result[2]
                    if is_warn_res0_ov == '0':
                        syscnt = result[4]
                    else:
                        syscnt = None
                else:
                    logger.info("Profiling: invalid hwts log record type %s", ms_type)
                    continue

                if int(task_id) < 25000:
                    task_id = str(stream_id) + "_" + str(task_id)
                result_data += ("%-14s %-4s %-8s %-9s %-8s %-15s %s\n" %(log_type[int(ms_type, 2)], cnt, core_id,
                                                                         blk_id, task_id, syscnt, stream_id))

        fwrite_format(self._output_filename, data_source=self._dst_file_title, is_start=True)
        fwrite_format(self._output_filename, data_source=self._dst_file_column_title)
        fwrite_format(self._output_filename, data_source=result_data)

        return True
