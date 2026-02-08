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
"""Ascend kernel details viewer"""
import os
from decimal import Decimal

from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.constant import (
    JitLevel,
    ProfilerLevel,
    OpSummaryHeaders,
    ProfilerActivity
)
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.file_manager import FileManager
from mindspore import log as logger


class AscendKernelDetailsViewer(BaseViewer):
    """
    Ascend kernel details viewer
    """
    KERNEL_DETAILS_FILE_NAME = "kernel_details.csv"
    EXCLUDE_HEADERS = [OpSummaryHeaders.DEVICE_ID.value]
    LEVEL0_EXCLUDE_HEADERS = [
        OpSummaryHeaders.MIX_BLOCK_DIM.value,
        OpSummaryHeaders.HF32_ELIGIBLE.value,
        OpSummaryHeaders.INPUT_SHAPES.value,
        OpSummaryHeaders.INPUT_DATA_TYPES.value,
        OpSummaryHeaders.INPUT_FORMATS.value,
        OpSummaryHeaders.OUTPUT_SHAPES.value,
        OpSummaryHeaders.OUTPUT_DATA_TYPES.value,
        OpSummaryHeaders.OUTPUT_FORMATS.value,
        OpSummaryHeaders.CONTEXT_ID.value,
    ]
    RENAME_HEADERS = {
        OpSummaryHeaders.OP_NAME.value: "Name",
        OpSummaryHeaders.OP_TYPE.value: "Type",
        OpSummaryHeaders.TASK_TYPE.value: "Accelerator Core",
        OpSummaryHeaders.TASK_START_TIME.value: "Start Time(us)",
        OpSummaryHeaders.TASK_DURATION.value: "Duration(us)",
        OpSummaryHeaders.TASK_WAIT_TIME.value: "Wait Time(us)",
    }

    def __init__(self, **kwargs):
        super().__init__()
        self._save_path = os.path.join(
            kwargs.get("ascend_profiler_output_path"),
            self.KERNEL_DETAILS_FILE_NAME
        )
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        self._is_set_schedule = kwargs.get("is_set_schedule")
        self._jit_level = kwargs.get("jit_level")
        self._profiler_level = kwargs.get("profiler_level")
        self._activities = kwargs.get("activities")
        self.op_summary_headers = None
        self.op_summary = None
        self.trace_container = None
        self.kernel_details_headers = None
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

    def save(self, data):
        """
        Save kernel details to csv file.
        """
        self._logger.info("AscendKernelDetailsViewer start")
        try:
            if self._profiler_level == ProfilerLevel.LevelNone.value:
                return
            self._check_input_data(data)
            self._update_kernel_name_and_step_id()
            self._update_headers()
            self._write_data()
            self._logger.info("Kernel details saved done")
        except Exception as e:  # pylint: disable=W0703
            self._logger.error("Failed to save kernel details: %s", str(e), exc_info=True)
        self._logger.info("AscendKernelDetailsViewer end")

    def _check_input_data(self, data):
        """
        Check input data.
        """
        self.trace_container = data.get("trace_view_container", None)
        self.op_summary = data.get("op_summary", None)
        self.op_summary_headers = data.get("op_summary_headers", None)

        if self.op_summary is None or self.op_summary.size == 0:
            raise ValueError("op summary is empty")

        if self.trace_container is None:
            raise ValueError("trace view container is None")

    def _write_data(self):
        """
        Write data to csv file.
        """
        self._logger.info("Kernel details saved start")
        csv_data = []
        for row in self.op_summary:
            csv_row = [row[field] for field in self.op_summary_headers]
            csv_data.append(csv_row)
        FileManager.create_csv_file(
            file_path=self._save_path,
            data=csv_data,
            headers=self.kernel_details_headers
        )
        self._logger.info("Kernel details saved done")

    def _update_headers(self):
        """
        Update kernel details headers.
        """
        # filter exclude headers
        self.op_summary_headers = [
            header
            for header in self.op_summary_headers
            if header not in self.EXCLUDE_HEADERS
        ]

        if self._profiler_level == ProfilerLevel.Level0.value:
            self.op_summary_headers = [
                header
                for header in self.op_summary_headers
                if header not in self.LEVEL0_EXCLUDE_HEADERS
            ]

        if (not self._is_set_schedule or self._jit_level == JitLevel.GRAPH_LEVEL or
                not self.trace_container.get_step_id_time_dict()):
            self.op_summary_headers.remove(OpSummaryHeaders.STEP_ID.value)

        # rename headers
        self.kernel_details_headers = [
            self.RENAME_HEADERS.get(header, header)
            for header in self.op_summary_headers
        ]

    def _update_kernel_name_and_step_id(self):
        """
        Update kernel op name to framework launch op name and step id.
        """
        self._logger.info("Update kernel name start")

        dev_kernels = self.trace_container.hardware_op_event
        step_id_to_time_dict = self.trace_container.get_step_id_time_dict()

        # activities parameter NPU+CPU、CPU
        if ProfilerActivity.CPU.value in self._activities:
            self._update_kernel_detail_op_name_and_step_id(dev_kernels, step_id_to_time_dict)

    def _update_kernel_detail_op_name_and_step_id(self, dev_kernels, step_id_to_time_dict):
        """
        Update op summary op name and step id in NPU+CPU、CPU scenes.
        """
        _generate_hardware_op_event_step_id(dev_kernels, step_id_to_time_dict)

        if not dev_kernels and self._jit_level != JitLevel.GRAPH_LEVEL:
            logger.warning(
                "Cannot find the device kernels with MindSpore framework launch op, "
            )
            return

        # build device kernel to framework launch op map
        dev_kernel_to_fwk_op = {}
        for _, per_tid_kernels in dev_kernels.items():
            for kernel in per_tid_kernels:
                dev_kernel_name = kernel.name
                dev_kerel_ts = str(kernel.ts)
                dev_kernel_to_fwk_op[(dev_kernel_name, dev_kerel_ts)] = kernel

        launch_ops = [None] * len(self.op_summary)
        step_ids = [None] * len(self.op_summary)
        for index, summary in enumerate(self.op_summary):
            dev_kernel_name = summary[OpSummaryHeaders.OP_NAME.value]
            dev_kernel_ts = str(summary[OpSummaryHeaders.TASK_START_TIME.value]).strip("\t")
            fwk_langch_op_name = None
            step_id = None
            if dev_kernel_to_fwk_op.get((dev_kernel_name, dev_kernel_ts)):
                kernel = dev_kernel_to_fwk_op.get((dev_kernel_name, dev_kernel_ts))
                if kernel.parent:
                    fwk_langch_op_name = kernel.parent.name
                step_id = kernel.step_id

            if step_id is None:
                step_id = _get_step_id_by_ts(Decimal(dev_kernel_ts), step_id_to_time_dict)

            if fwk_langch_op_name is None:
                self._logger.warning(
                    "Can not find fwk launch op for dev kernel %s, ts %s",
                    dev_kernel_name,
                    dev_kernel_ts,
                )
                launch_ops[index] = dev_kernel_name
            else:
                launch_ops[index] = f"{fwk_langch_op_name}/{dev_kernel_name}"

            if step_id is None and self._is_set_schedule and self._jit_level != JitLevel.GRAPH_LEVEL:
                self._logger.warning(
                    "Can not find step id for dev kernel %s, ts %s",
                    dev_kernel_name,
                    dev_kernel_ts,
                )
            else:
                step_ids[index] = step_id

        # update op summary op name
        self.op_summary[OpSummaryHeaders.OP_NAME.value] = launch_ops

        # update op summary step id
        if self._is_set_schedule and self._jit_level != JitLevel.GRAPH_LEVEL:
            self.op_summary[OpSummaryHeaders.STEP_ID.value] = step_ids

        self._logger.info("Update kernel name done")


def _generate_hardware_op_event_step_id(hardware_op_events_dict: dict, step_id_to_time_dict: dict):
    """
    Generate the hardware op event step id.
    """
    for hardware_op_events_list in hardware_op_events_dict.values():
        # Associate each hardware operation event with its step ID
        for hardware_op_event in hardware_op_events_list:
            kernel_event = hardware_op_event.parent
            if not kernel_event:
                continue

            hardware_op_event.step_id = _get_step_id_by_ts(kernel_event.ts, step_id_to_time_dict)


def _get_step_id_by_ts(ts: Decimal, step_events_dict: dict):
    """
    Retrieves the step ID for a given timestamp from the step events dictionary.
    """
    # Iterate through the step events dictionary to find the step ID for the given timestamp
    for step_id, (st, et) in step_events_dict.items():
        if st <= ts <= et:
            return step_id

    if step_events_dict:
        return list(step_events_dict.keys())[-1]

    return None
