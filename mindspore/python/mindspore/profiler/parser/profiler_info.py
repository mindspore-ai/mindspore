# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Record profiler information"""
import json
import os
import stat

from mindspore.version import __version__ as ms_version
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path


class ProfilerInfo:
    """
    This class is used to record profiler information.
    it contains context_mode, rank_id, rank_size, parallel_mode, pipeline_stage_num, pipeline_stage_id,
    profiling_start_time, profiling_stop_time, analyse_start_time, analyse_end_time
    """

    _file_name = "profiler_info_{}.json"
    _file_path = ""
    _profiler_info_dict = dict()

    @staticmethod
    def init_info(context_mode, rank_id):
        """Profiler info initialization must include context_mode, rank_id and output_path."""
        ProfilerInfo._profiler_info_dict["context_mode"] = context_mode
        ProfilerInfo._profiler_info_dict["rank_id"] = rank_id
        ProfilerInfo._profiler_info_dict["ms_version"] = ms_version
        ProfilerInfo._file_name = ProfilerInfo._file_name.format(rank_id)

    @staticmethod
    def set_parallel_info(parallel_mode="", pipeline_stage_num=1, pipeline_stage_id=0):
        """Set parallel info include parallel_mode, pipeline_stage_num and pipeline_stage_id."""
        info = dict()
        info["parallel_mode"] = parallel_mode
        info["pipeline_stage_num"] = pipeline_stage_num
        info["pipeline_stage_id"] = pipeline_stage_id
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_profiling_start_time(start_time):
        """Set the profiling start time."""
        info = dict()
        info["profiling_start_time"] = start_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_profiling_stop_time(stop_time):
        """Set the profiling stop time."""
        info = dict()
        info["profiling_stop_time"] = stop_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_analyse_start_time(start_time):
        """Set the analyse start time."""
        info = dict()
        info["analyse_start_time"] = start_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_analyse_end_time(end_time):
        """Set the analyse end time."""
        info = dict()
        info["analyse_end_time"] = end_time
        ProfilerInfo._profiler_info_dict.update(info)

    @staticmethod
    def set_graph_ids(graph_ids):
        """Set the graph id list."""
        ProfilerInfo._profiler_info_dict["graph_ids"] = graph_ids

    @staticmethod
    def set_rank_size(rank_size):
        """Set the rank size."""
        ProfilerInfo._profiler_info_dict["rank_size"] = rank_size

    @staticmethod
    def set_heterogeneous(is_heterogeneous):
        """Set is it heterogeneous."""
        ProfilerInfo._profiler_info_dict["is_heterogeneous"] = is_heterogeneous

    @staticmethod
    def get_profiler_info():
        """Get the profiler info."""
        return ProfilerInfo._profiler_info_dict

    @staticmethod
    def save(output_path):
        """Save the profiler info to file."""
        ProfilerInfo._file_path = os.path.join(output_path, ProfilerInfo._file_name)
        ProfilerInfo._file_path = validate_and_normalize_path(ProfilerInfo._file_path)
        with os.fdopen(os.open(ProfilerInfo._file_path,
                               os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as json_file:
            json.dump(ProfilerInfo._profiler_info_dict, json_file)
        os.chmod(ProfilerInfo._file_path, stat.S_IREAD | stat.S_IWRITE)
