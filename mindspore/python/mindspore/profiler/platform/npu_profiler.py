# Copyright 2024 Huawei Technologies Co., Ltd
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
"""NPU platform profiler."""
import os
import glob
import json
from typing import List, Optional

from mindspore import log as logger
import mindspore._c_dataengine as cde
import mindspore._c_expression as c_expression

from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.registry import PROFILERS
from mindspore.profiler.common.constant import (
    DeviceTarget,
    ProfilerActivity,
    AnalysisMode,
    ExportType,
)
from mindspore._c_expression import _framework_profiler_enable_mi, _framework_profiler_disable_mi
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.platform.base_profiler import BaseProfiler
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager
from mindspore.profiler.common.profiler_info import ProfilerInfo
from mindspore.profiler.common.process_pool import MultiProcessPool
from mindspore.profiler.common.constant import MsprofModeName
from mindspore.profiler.common.util import no_exception_func
from mindspore.profiler.analysis.task_manager import TaskManager
from mindspore.profiler.analysis.time_converter import TimeConverter
from mindspore.profiler.analysis.parser.ascend_cann_parser import AscendMsprofParser
from mindspore.profiler.analysis.parser.ms_framework_parser import FrameworkParser
from mindspore.profiler.analysis.parser.ms_minddata_parser import MindDataParser
from mindspore.profiler.analysis.parser.framework_cann_relation_parser import FrameworkCannRelationParser
from mindspore.profiler.analysis.viewer.ms_dataset_viewer import MsDatasetViewer
from mindspore.profiler.analysis.viewer.ascend_timeline_viewer import AscendTimelineViewer
from mindspore.profiler.analysis.viewer.ascend_kernel_details_viewer import AscendKernelDetailsViewer
from mindspore.profiler.analysis.viewer.ascend_step_trace_time_viewer import AscendStepTraceTimeViewer
from mindspore.profiler.analysis.viewer.ascend_communication_viewer import AscendCommunicationViewer
from mindspore.profiler.analysis.viewer.ascend_integrate_viewer import AscendIntegrateViewer
from mindspore.profiler.analysis.viewer.ascend_memory_viewer import AscendMemoryViewer
from mindspore.profiler.analysis.viewer.ascend_op_memory_viewer import AscendOpMemoryViewer
from mindspore.profiler.analysis.viewer.ms_minddata_viewer import (
    MindDataPipelineRawViewer,
    MindDataPiplineSummaryViewer,
)
from mindspore.profiler.analysis.viewer.ms_operator_details_viewer import MsOperatorDetailsViewer
from mindspore.profiler.common.util import print_msg_with_pid
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.mstx import Mstx
from mindspore.profiler.common.util import get_device_id


@PROFILERS.register_module(DeviceTarget.NPU.value)
class NpuProfiler(BaseProfiler):
    """
    NPU platform profiler
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_env_not_valid = self._is_environment_not_valid()
        self._prof_ctx = ProfilerContext()
        self._prof_info = ProfilerInfo()
        self._prof_path_mgr = ProfilerPathManager()
        ProfilerLogger.init(self._prof_ctx.ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

        self._profiler = c_expression.Profiler.get_instance(DeviceTarget.NPU.value)
        # initialize profiler backend
        self._profiler.init(
            self._prof_ctx.ascend_ms_dir,
            int(get_device_id()),
            json.dumps(self._prof_ctx.npu_profiler_params),
        )
        self._logger.info("NpuProfiler init profiler backend params %s",
                          json.dumps(self._prof_ctx.npu_profiler_params, indent=4))

        # record original profiler params
        self._prof_info.profiler_parameters = self._prof_ctx.original_params

        # initialize minddata profiler
        if self._prof_ctx.data_process:
            self._md_profiler = cde.GlobalContext.profiling_manager()
            self._md_profiler.init()
            self._logger.info("NpuProfiler init minddata profiler")

        self._prof_mgr = c_expression.ProfilerManager.get_instance()

    def start(self) -> None:
        """Start profiling."""
        if self._is_env_not_valid:
            return
        self._logger.info("NpuProfiler start.")

        Mstx.enable = self._prof_ctx.npu_profiler_params.get("mstx", False)

        if ProfilerActivity.CPU in self._prof_ctx.activities:
            _framework_profiler_enable_mi()
            self._prof_mgr.set_profile_framework("time")
            self._logger.info("NpuProfiler start enable framework")

        if self._profiler:
            self._profiler.start()

        if self._prof_ctx.data_process:
            self._md_profiler.start()
            self._logger.info("NpuProfiler start minddata profiler")

    def stop(self) -> None:
        """Stop profiling."""
        if self._is_env_not_valid:
            return
        self._logger.info("NpuProfiler stop.")

        Mstx.enable = False

        if self._profiler:
            self._profiler.stop()

        if ProfilerActivity.CPU in self._prof_ctx.activities:
            _framework_profiler_disable_mi()
            self._prof_mgr.set_profile_framework("NULL")
            self._logger.info("NpuProfiler stop disable framework")

        if self._prof_ctx.data_process:
            self._md_profiler.stop()
            self._md_profiler.save(self._prof_ctx.framework_path)

        if ProfilerActivity.NPU in self._prof_ctx.activities:
            prof_dir = glob.glob(os.path.join(self._prof_ctx.ascend_ms_dir, "PROF_*"))
            if not prof_dir:
                logger.error(f"No PROF_* directory found in {self._prof_ctx.ascend_ms_dir}")
                return

            self._prof_ctx.msprof_profile_path = prof_dir[0]
            self._prof_ctx.device_id = self._prof_ctx.msprof_profile_device_path.split("_")[-1]

        self._prof_info.ms_profiler_info = {
            "rank_id": self._prof_ctx.rank_id,
            "device_id": self._prof_ctx.device_id,
        }

        self._prof_info.save(self._prof_ctx.ascend_ms_dir, self._prof_ctx.rank_id)

    def analyse(self, **kwargs) -> None:
        """Analyse the profiling data."""
        if self._is_env_not_valid:
            return
        self._logger.info("NpuProfiler analyse.")

        NPUProfilerAnalysis.online_analyse(async_mode=kwargs.get('async_mode'))

    def finalize(self) -> None:
        """Finalize profiling data."""
        if self._is_env_not_valid:
            return
        self._logger.info("NpuProfiler finalize.")
        if self._profiler:
            self._profiler.finalize()

    @staticmethod
    def _is_environment_not_valid() -> bool:
        # check msprof dynamic environment variable
        if os.getenv(MsprofModeName.MSPROF_DYNAMIC_ENV) is not None:
            logger.error(f"The environment variable '{MsprofModeName.MSPROF_DYNAMIC_ENV}' has been set."
                         f"Please execute 'unset {MsprofModeName.MSPROF_DYNAMIC_ENV}'.")
            return True
        return False


class NPUProfilerAnalysis:
    """
    NPU profiler analysis interface
    """

    @classmethod
    @no_exception_func()
    def online_analyse(cls, async_mode: bool = False):
        """
        Online analysis for NPU
        """
        cls._pre_analyse_online()
        if async_mode:
            ProfilerContext().mode = AnalysisMode.ASYNC_MODE.value
            MultiProcessPool().add_async_job(cls._run_tasks, **ProfilerContext().to_dict())
        else:
            ProfilerContext().mode = AnalysisMode.SYNC_MODE.value
            cls._run_tasks(**ProfilerContext().to_dict())

    @classmethod
    @no_exception_func()
    def offline_analyse(
            cls,
            path: str,
            pretty: bool,
            step_list: Optional[List[int]],
            data_simplification: bool,
    ) -> None:
        """Analyze profiling data in offline mode."""
        ProfilerLogger.init(path)
        cls._pre_analyse_offline(path, pretty, step_list, data_simplification)
        cls._run_tasks(**ProfilerContext().to_dict())

    @classmethod
    def _pre_analyse_online(cls):
        """
        Pre-process for online analysis
        """
        prof_ctx = ProfilerContext()
        if ProfilerActivity.NPU in prof_ctx.activities:
            ProfilerPathManager().clean_analysis_cache()
            ProfilerPathManager().create_output_path()
            ProfilerInfo().load_time_parameters(
                prof_ctx.msprof_profile_path, prof_ctx.msprof_profile_host_path
            )
            TimeConverter.init_parameters(**ProfilerInfo().time_parameters)

        elif prof_ctx.activities == [ProfilerActivity.CPU]:
            ProfilerPathManager().create_output_path()
            TimeConverter.init_parameters(freq=100.0, cntvct=0, localtime_diff=0)

    @classmethod
    def _pre_analyse_offline(
            cls,
            ascend_ms_dir: str,
            pretty: bool,
            step_list: Optional[List[int]],
            data_simplification: bool,
    ) -> None:
        """Pre-process profiling data for offline analysis."""
        prof_ctx = ProfilerContext()
        prof_info = ProfilerInfo()
        prof_info_file_path = PathManager.get_profiler_info_path(ascend_ms_dir)
        prof_info.load_info(prof_info_file_path)
        prof_ctx.device_id = prof_info.ms_profiler_info["device_id"]
        prof_ctx.rank_id = prof_info.ms_profiler_info["rank_id"]
        prof_ctx.set_params()
        prof_ctx.load_offline_profiler_params(prof_info.profiler_parameters)
        prof_ctx.jit_level = prof_info.jit_level

        if ProfilerActivity.NPU in prof_ctx.activities:
            prof_dir = glob.glob(os.path.join(ascend_ms_dir, "PROF_*"))
            if not prof_dir:
                logger.error(f"No PROF_* directory found in {ascend_ms_dir}")
                return
            prof_path_mgr = ProfilerPathManager()

            # set PROF_XXX path
            prof_ctx.ascend_ms_dir = ascend_ms_dir
            prof_ctx.msprof_profile_path = prof_dir[0]
            prof_info.load_time_parameters(
                prof_ctx.msprof_profile_path, prof_ctx.msprof_profile_host_path
            )
            prof_ctx.pretty = pretty
            prof_ctx.step_list = step_list
            prof_ctx.data_simplification = data_simplification
            prof_path_mgr.clean_analysis_cache()
            prof_path_mgr.create_output_path()
            TimeConverter.init_parameters(**prof_info.time_parameters)
        elif [ProfilerActivity.CPU] == prof_ctx.activities:
            prof_ctx.ascend_ms_dir = ascend_ms_dir
            ProfilerPathManager().create_output_path()
            TimeConverter.init_parameters(freq=100.0, cntvct=0, localtime_diff=0)

    @classmethod
    def _run_tasks(cls, **kwargs) -> None:
        """
        Run tasks for online analysis
        """
        ascend_ms_dir = kwargs.get("ascend_ms_dir", "")
        print_msg_with_pid(f"Start parsing profiling data in {kwargs.get('mode')} mode at: {ascend_ms_dir}")
        task_mgr = cls._construct_task_mgr(**kwargs)
        task_mgr.run()
        ProfilerLogger.get_instance().info(json.dumps(task_mgr.cost_time, indent=4))
        activities = kwargs.get("activities", [])
        export_type = kwargs.get("export_type", [])
        if ProfilerActivity.NPU.value in activities:
            if ExportType.Db.value in export_type:
                ProfilerPathManager().move_db_file()
            else:
                ProfilerPathManager().remove_db_file()
            if kwargs.get("data_simplification"):
                ProfilerPathManager().simplify_data()

    @classmethod
    def _construct_task_mgr(cls, **kwargs) -> TaskManager:
        """
        Construct task manager based on activities and parameters
        """
        task_mgr = TaskManager()
        activities = kwargs.get("activities", [])
        export_type = kwargs.get("export_type", [])
        record_shapes = kwargs.get("record_shapes", False)
        enable_data_process = kwargs.get("data_process", False)

        # CANN flow parser
        cann_flow_parsers = []

        if export_type == [ExportType.Db.value]:
            if ProfilerActivity.NPU.value in activities:
                cann_flow_parsers.append(
                    AscendMsprofParser(**kwargs)
                )
                task_mgr.create_flow(
                    *cann_flow_parsers, flow_name="cann_flow", show_process=True
                )
            return task_mgr

        if ProfilerActivity.NPU.value in activities:
            cann_flow_parsers.append(
                AscendMsprofParser(**kwargs)
                .register_post_hook(AscendMemoryViewer(**kwargs).save)
                .register_post_hook(AscendOpMemoryViewer(**kwargs).save)
            )

        if ProfilerActivity.CPU.value in activities:
            cann_flow_parsers.append(
                FrameworkParser(**kwargs).register_post_hook(
                    MsDatasetViewer(**kwargs).save
                )
            )
            if record_shapes:
                cann_flow_parsers.append(
                    FrameworkCannRelationParser(**kwargs).register_post_hook(
                        MsOperatorDetailsViewer(**kwargs).save
                    )
                )

        if ProfilerActivity.NPU.value in activities:
            cann_flow_parsers.append(
                FrameworkCannRelationParser(**kwargs)
                .register_post_hook(AscendTimelineViewer(**kwargs).save)
                .register_post_hook(AscendKernelDetailsViewer(**kwargs).save)
                .register_post_hook(AscendStepTraceTimeViewer(**kwargs).save)
                .register_post_hook(AscendIntegrateViewer(**kwargs).save)
                .register_post_hook(AscendCommunicationViewer(**kwargs).save)
            )
        elif ProfilerActivity.CPU.value in activities:
            cann_flow_parsers.append(
                FrameworkCannRelationParser().register_post_hook(
                    AscendTimelineViewer(**kwargs).save
                )
            )

        task_mgr.create_flow(
            *cann_flow_parsers, flow_name="cann_flow", show_process=True
        )

        # MindData flow parser
        if enable_data_process:
            task_mgr.create_flow(
                MindDataParser(**kwargs)
                .register_post_hook(MindDataPipelineRawViewer(**kwargs).save)
                .register_post_hook(MindDataPiplineSummaryViewer(**kwargs).save),
                flow_name="minddata_flow",
                show_process=False,
            )

        return task_mgr
