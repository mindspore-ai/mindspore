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
"""Profiler Context"""
import os
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Set,
    Callable,
)

from mindspore.communication.management import GlobalComm
from mindspore.communication.management import get_rank
from mindspore.profiler.common.constant import (
    DeviceTarget,
    ProfilerLevel,
    ProfilerActivity,
    AicoreMetrics,
    ExportType,
    HostSystem
)
from mindspore.profiler.common.profiler_output_path import ProfilerOutputPath
from mindspore.profiler.common.profiler_parameters import ProfilerParameters
from mindspore.profiler.common.constant import AnalysisMode
from mindspore.profiler.common.singleton import Singleton
from mindspore.profiler.schedule import Schedule

from mindspore import context
from mindspore import log as logger
from mindspore.profiler.common.profiler_info import ProfilerInfo
from mindspore.profiler.experimental_config import _ExperimentalConfig
from mindspore.profiler.common.util import get_device_id


@Singleton
class ProfilerContext:
    """
    Profiler context manage all parameters and paths on runtime.
    """

    def __init__(self):
        self._profiler_params_mgr: ProfilerParameters = None
        self._device_id: Optional[str] = None
        self._rank_id: Optional[str] = None
        self._device_target: Optional[str] = None
        self._dynamic_status: Optional[bool] = None
        self._step_list: Optional[int] = None
        self._mode: str = AnalysisMode.SYNC_MODE.value
        self._pretty: bool = False
        self._profiler_path_mgr: ProfilerOutputPath = None
        self._on_trace_ready_output_path = None
        self._jit_level: Optional[str] = ""

        self._init_device_target()
        self._init_device_id()
        self._init_rank_id()
        self._init_jit_level()

    def set_params(self, **kwargs):
        """
        Set profiler parameters and paths
        """
        # output_path and on_trace_ready cannot be set at the same time. If both are set,
        # only paths in on_trace_ready take effect
        if self._on_trace_ready_output_path:
            final_path = self._on_trace_ready_output_path
            if "output_path" in kwargs:
                logger.warning(f"Both on_trace_ready path and output_path are provided. "
                               f"The on_trace_ready path takes effect. Final path is {final_path}")
            kwargs["output_path"] = final_path
        if kwargs.get("experimental_config"):
            self._check_and_set_experimental_params(kwargs)
        self._profiler_params_mgr: ProfilerParameters = ProfilerParameters(**kwargs)
        self._profiler_path_mgr: ProfilerOutputPath = ProfilerOutputPath(rank_id=int(self._rank_id))
        self._profiler_path_mgr.output_path = self._profiler_params_mgr.output_path

    @staticmethod
    def _check_and_set_experimental_params(kwargs):
        """
        Set experimental parameters
        """
        if not isinstance(kwargs.get("experimental_config"), _ExperimentalConfig):
            logger.warning("For Profiler, experimental_config value must be the "
                           "'mindspore.profiler._ExperimentalConfig' class, "
                           "reset to default value.")
            return
        kwargs["profiler_level"] = kwargs.get("experimental_config").profiler_level
        kwargs["aic_metrics"] = kwargs.get("experimental_config").aic_metrics
        kwargs["l2_cache"] = kwargs.get("experimental_config").l2_cache
        kwargs["mstx"] = kwargs.get("experimental_config").mstx
        kwargs["data_simplification"] = kwargs.get("experimental_config").data_simplification
        kwargs["export_type"] = kwargs.get("experimental_config").export_type
        kwargs["mstx_domain_include"] = kwargs.get("experimental_config").mstx_domain_include
        kwargs["mstx_domain_exclude"] = kwargs.get("experimental_config").mstx_domain_exclude
        kwargs["sys_io"] = kwargs.get("experimental_config").sys_io
        kwargs["sys_interconnection"] = kwargs.get("experimental_config").sys_interconnection
        kwargs["host_sys"] = kwargs.get("experimental_config").host_sys

    @property
    def on_trace_ready_output_path(self) -> str:
        """Get the on trace ready output path."""
        return self._on_trace_ready_output_path

    @on_trace_ready_output_path.setter
    def on_trace_ready_output_path(self, value: str):
        """Set the tensorboard profile path to on trace ready output path."""
        self._on_trace_ready_output_path = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profiler context to a dictionary for multiprocessing.
        """
        return {
            **self._profiler_params_mgr.original_params,
            **self._profiler_path_mgr.to_dict(),
            "device_id": self._device_id,
            "rank_id": self._rank_id,
            "device_target": self._device_target,
            "dynamic_status": self._dynamic_status,
            "step_list": self._step_list,
            "mode": self._mode,
            "jit_level": self._jit_level,
        }

    def load_offline_profiler_params(self, profiler_parameters: Dict[str, Any]) -> None:
        """
        Update profiler parameters from profiler_info.json
        """
        if not profiler_parameters:
            raise ValueError("Profiler parameters is empty")

        for param, (_, _) in self._profiler_params_mgr.PARAMS.items():
            if param in profiler_parameters:
                if param == "profiler_level":
                    value = ProfilerLevel(profiler_parameters[param])
                elif param == "aic_metrics":
                    value = AicoreMetrics(profiler_parameters[param])
                elif param == "activities":
                    value = [ProfilerActivity(activity) for activity in profiler_parameters[param]]
                elif param == "export_type":
                    value = [ExportType(export_type) for export_type in profiler_parameters[param]]
                elif param == "host_sys":
                    value = [HostSystem(host_sys) for host_sys in profiler_parameters[param]]
                elif param == "schedule":
                    continue
                else:
                    value = profiler_parameters[param]

                setattr(self._profiler_params_mgr, param, value)
        setattr(self._profiler_params_mgr, "is_set_schedule", profiler_parameters["is_set_schedule"])

    @property
    def device_target_set(self) -> Set[str]:
        """
        Get the device target set for ProfilerInterface initialization.

        CPU is always included in the list, device_target includes CPU、Ascend、GPU.
        """
        return set([DeviceTarget.CPU.value, self._device_target])

    @property
    def npu_profiler_params(self) -> Dict[str, Any]:
        """
        Get NPU profiler parameters for Ascend profiler cpp backend.

        Returns:
            Dict[str, Any]: A dictionary of NPU profiler parameters.
        """
        params = self._profiler_params_mgr.npu_profiler_params
        # update framework_path for profile memory
        params["framework_path"] = self._profiler_path_mgr.framework_path
        params["rank_id"] = int(self._rank_id)
        params["device_id"] = int(self._device_id)
        return params

    @property
    def original_params(self) -> Dict[str, str]:
        """Get the original parameters from ProfilerParameters."""
        return self._profiler_params_mgr.original_params

    @property
    def output_path(self) -> str:
        """Get the output path from ProfilerOutputPath."""
        return self._profiler_path_mgr.output_path

    @property
    def ascend_ms_dir(self) -> str:
        """Get the Ascend MS directory from ProfilerOutputPath."""
        return self._profiler_path_mgr.ascend_ms_dir

    @ascend_ms_dir.setter
    def ascend_ms_dir(self, value: str):
        """Set the Ascend MS directory to ProfilerOutputPath."""
        self._profiler_path_mgr.ascend_ms_dir = value

    @property
    def ascend_profiler_output_path(self) -> str:
        """Get the Ascend profiler output path from ProfilerOutputPath."""
        return self._profiler_path_mgr.ascend_profiler_output_path

    @property
    def framework_path(self) -> str:
        """Get the framework path from ProfilerOutputPath."""
        return self._profiler_path_mgr.framework_path

    @property
    def msprof_profile_path(self) -> str:
        """Get the MSProf profile path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_path

    @msprof_profile_path.setter
    def msprof_profile_path(self, value: str):
        """Set the MSProf profile path to ProfilerOutputPath."""
        self._profiler_path_mgr.msprof_profile_path = value

    @property
    def msprof_profile_host_path(self) -> str:
        """Get the MSProf profile host path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_host_path

    @property
    def msprof_profile_device_path(self) -> str:
        """Get the MSProf profile device path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_device_path

    @property
    def msprof_profile_log_path(self) -> str:
        """Get the MSProf profile log path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_log_path

    @property
    def msprof_profile_output_path(self) -> str:
        """Get the MSProf profile output path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_output_path

    @property
    def profiler_level(self) -> ProfilerLevel:
        """Get the profiler level from ProfilerParameters."""
        return self._profiler_params_mgr.profiler_level

    @property
    def activities(self) -> List[ProfilerActivity]:
        """Get the activities from ProfilerParameters."""
        return self._profiler_params_mgr.activities

    @property
    def profile_memory(self) -> bool:
        """Get the profile memory from ProfilerParameters."""
        return self._profiler_params_mgr.profile_memory

    @property
    def parallel_strategy(self) -> bool:
        """Get the parallel strategy from ProfilerParameters."""
        return self._profiler_params_mgr.parallel_strategy

    @property
    def start_profile(self) -> bool:
        """Get the start profile from ProfilerParameters."""
        return self._profiler_params_mgr.start_profile

    @property
    def aicore_metrics(self) -> int:
        """Get the aicore metrics from ProfilerParameters."""
        return self._profiler_params_mgr.aicore_metrics

    @property
    def l2_cache(self) -> bool:
        """Get the l2 cache from ProfilerParameters."""
        return self._profiler_params_mgr.l2_cache

    @property
    def hbm_ddr(self) -> bool:
        """Get the hbm ddr from ProfilerParameters."""
        return self._profiler_params_mgr.hbm_ddr

    @property
    def pcie(self) -> bool:
        """Get the pcie from ProfilerParameters."""
        return self._profiler_params_mgr.pcie

    @property
    def sync_enable(self) -> bool:
        """Get the sync enable from ProfilerParameters."""
        return self._profiler_params_mgr.sync_enable

    @property
    def data_process(self) -> bool:
        """Get the data process from ProfilerParameters."""
        return self._profiler_params_mgr.data_process

    @property
    def with_stack(self) -> bool:
        """Get the with stack from ProfilerParameters."""
        return self._profiler_params_mgr.with_stack

    @property
    def mstx(self) -> bool:
        """Get the mstx from ProfilerParameters."""
        return self._profiler_params_mgr.mstx

    @property
    def mstx_domain_include(self) -> List[str]:
        """Get the mstx domain include from ProfilerParameters."""
        return self._profiler_params_mgr.mstx_domain_include

    @property
    def mstx_domain_exclude(self) -> List[str]:
        """Get the mstx domain exclude from ProfilerParameters."""
        return self._profiler_params_mgr.mstx_domain_exclude

    @property
    def data_simplification(self) -> bool:
        """Get the data simplification from ProfilerParameters."""
        return self._profiler_params_mgr.data_simplification

    @data_simplification.setter
    def data_simplification(self, value: bool) -> None:
        """Set data simplification value."""
        if not isinstance(value, bool):
            logger.warning(f"For profiler, the type of data_simplification should be bool, "
                           f"but got {type(value)}, reset to True.")
            value = True
        self._profiler_params_mgr.data_simplification = value

    @property
    def record_shapes(self) -> bool:
        """Get the record shapes from ProfilerParameters."""
        return self._profiler_params_mgr.record_shapes

    @property
    def device_target(self) -> str:
        """Get device target."""
        return self._device_target

    @property
    def rank_id(self) -> str:
        """Get rank id."""
        return self._rank_id

    @rank_id.setter
    def rank_id(self, value: str) -> None:
        """Set rank id."""
        if not value:
            raise ValueError("Rank id must be a non-empty string")

        if not value.isdigit():
            raise ValueError("Rank id must be a number")
        self._rank_id = value

    @property
    def device_id(self) -> str:
        """Get device id."""
        return self._device_id

    @device_id.setter
    def device_id(self, value) -> None:
        """Set device id."""
        if not value:
            raise ValueError("Device id must be a non-empty string")

        if not value.isdigit():
            raise ValueError("Device id must be a number")
        self._device_id = value

    @property
    def dynamic_status(self) -> bool:
        """Get dynamic status."""
        return self._dynamic_status

    @property
    def step_list(self) -> Optional[List[int]]:
        """Get step list."""
        return self._step_list

    @step_list.setter
    def step_list(self, value: Optional[List[int]]) -> None:
        """Set step list for profiling."""
        if value is not None and not isinstance(value, list):
            logger.error(f"For profiler, the parameter step_list must be a list, "
                         f"but got type {type(value)}, step_list reset to None.")
            return
        if value:
            if not all(isinstance(step_id, int) for step_id in value):
                logger.error(f"For profiler, the elements of the parameter step_list "
                             "must be integers, step_list reset to None.")
                return
            value.sort()
            if value[-1] - value[0] != len(value) - 1:
                logger.error(f"For profiler, the elements of the parameter step_list "
                             "must be continuous integers, step_list reset to None.")
                return
        self._step_list = value

    @property
    def pretty(self) -> bool:
        return self._pretty

    @pretty.setter
    def pretty(self, value: bool) -> None:
        """Set pretty print value."""
        if not isinstance(value, bool):
            logger.warning(f"For profiler, the parameter pretty must be bool, "
                           f"but got {type(value)}, reset to False.")
            value = False
        self._pretty = value

    @property
    def mode(self) -> str:
        """Get the analysis mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Set analysis mode value."""
        if not isinstance(mode, str):
            logger.warning(f"For Profiler.analyse(), the parameter mode must be str, "
                           f"but got {type(mode)}, reset to {AnalysisMode.SYNC_MODE.value}.")
            self._mode = AnalysisMode.SYNC_MODE.value
            return

        mode_range = [m.value for m in AnalysisMode]
        if mode not in mode_range:
            logger.warning(f"For Profiler.analyse(), the parameter mode must be one of {mode_range}, "
                           f"but got {mode}, reset to {AnalysisMode.SYNC_MODE.value}.")
            self._mode = AnalysisMode.SYNC_MODE.value
            return

        self._mode = mode

    @property
    def schedule(self) -> Schedule:
        """Get the schedule from ProfilerParameters."""
        return self._profiler_params_mgr.schedule

    @property
    def on_trace_ready(self) -> Optional[Callable[..., Any]]:
        """Get the on trace ready from ProfilerParameters."""
        return self._profiler_params_mgr.on_trace_ready

    @property
    def is_set_schedule(self) -> bool:
        """Get the is set schedule from ProfilerParameters."""
        return self._profiler_params_mgr.is_set_schedule

    @property
    def jit_level(self) -> str:
        return self._jit_level

    @jit_level.setter
    def jit_level(self, value: str) -> None:
        """Set jit level value."""
        if not isinstance(value, str):
            logger.warning(f"For profiler, the parameter jit_level must be str, "
                           f"but got {type(value)}, reset to ''.")
            value = ""
        self._jit_level = value

    def _init_device_target(self) -> None:
        """
        Initialize the device target.

        Raises:
            RuntimeError: If the device target is not supported.
        """
        self._device_target = context.get_context("device_target")

        if self._device_target and self._device_target not in (
                member.value for member in DeviceTarget
        ):
            msg = "Profiling: unsupported backend: %s" % self._device_target
            raise RuntimeError(msg)

    def _init_device_id(self) -> None:
        """
        Initialize the device ID.
        """
        self._device_id = get_device_id()

    def _init_rank_id(self) -> None:
        """
        Initialize the rank ID.
        """
        if GlobalComm.INITED and self._device_target == DeviceTarget.NPU.value:
            self._rank_id = str(get_rank())
        else:
            self._rank_id = os.getenv("RANK_ID")

        if not self._rank_id or not self._rank_id.isdigit():
            self._rank_id = "0"

    def _init_jit_level(self):
        """
        Initialize the jit level.
        """
        jit_config = context.get_jit_config()
        self._jit_level = jit_config.get("jit_level", "")
        ProfilerInfo().jit_level = self._jit_level
