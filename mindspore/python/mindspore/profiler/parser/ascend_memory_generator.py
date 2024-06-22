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
"""memory data analyze module"""
import os

from decimal import Decimal

from mindspore.profiler.parser.ascend_analysis.file_manager import FileManager


class AscendMemoryGenerator:
    """Parsing the memory data of the ascend device"""
    GE_MEMORY_RECORD_HEADERS = ["Device id", "Component", "Timestamp(us)", "Total Allocated(KB)",
                                "Total Reserved(KB)", "Device"]
    MS_MEMORY_RECORD_HEADERS = ["Timestamp(ns)", "Total Allocated(Byte)", "Total Reserved(Byte)", "Total Active(Byte)"]
    TARGET_MEMORY_RECORD_HEADERS = ["Component", "Timestamp(us)", "Total Allocated(MB)", "Total Reserved(MB)",
                                    "Total Active(MB)", "Device Type"]

    def __init__(self, source_path: str, rank_id: int, mindstudio_profiler_output: str, ascend_ms_output_path: str):
        self._source_path = source_path
        self._rank_id = rank_id
        self._mindstudio_profiler_output = mindstudio_profiler_output
        self._ascend_ms_output_path = ascend_ms_output_path
        self._ge_memory_record = []
        self._ms_memory_record = []

    def parse(self):
        """Parse all memory data"""
        self.copy_npu_module_mem_csv()
        self.parse_memory_record()

    def copy_npu_module_mem_csv(self):
        """Generate npu_module_mem.csv"""
        npu_module_mem_file_list = FileManager.get_csv_file_list_by_start_name(self._mindstudio_profiler_output,
                                                                               "npu_module_mem")
        target_file_path = os.path.join(self._ascend_ms_output_path, "npu_module_mem.csv")
        FileManager.combine_csv_file(npu_module_mem_file_list, target_file_path)

    def parse_memory_record(self):
        """Generate memory_record.csv"""
        self._parse_ge_memory_record()
        self._parse_ms_memory_record()
        combined_memory_data = self._combine_ge_ms_memory_record()
        target_file_path = os.path.join(self._ascend_ms_output_path, "memory_record.csv")
        FileManager.create_csv_file(target_file_path, combined_memory_data, self.TARGET_MEMORY_RECORD_HEADERS)

    def _parse_ge_memory_record(self):
        """Parse ge memory record data"""
        memory_record_file_list = FileManager.get_csv_file_list_by_start_name(self._mindstudio_profiler_output,
                                                                              "memory_record")
        for file in memory_record_file_list:
            data = FileManager.read_csv_file(file)
            if len(data) > 1:
                self._ge_memory_record.extend(data[1:])

    def _parse_ms_memory_record(self):
        """Parse mindspore memory record data"""
        memory_record_file = os.path.join(self._source_path, f"cpu_ms_memory_record_{self._rank_id}.txt")
        data = FileManager.read_txt_file(memory_record_file)
        if len(data) > 1:
            self._ms_memory_record.extend(data[1:])

    def _get_app_reserved_memory(self) -> list:
        """Get the reserved memory of the application from npu_mem.csv"""
        npu_module_mem_file_list = FileManager.get_csv_file_list_by_start_name(self._mindstudio_profiler_output,
                                                                               "npu_mem")
        app_mems = []
        for file in npu_module_mem_file_list:
            md_mems = FileManager.read_csv_file(file)
            for mem in md_mems:
                if mem[1] == "APP":
                    app_mems.append(MemoryRecordBean([mem[1], mem[-1].rstrip('\t'), 0.0, float(mem[4]), 0.0,
                                                      f"NPU:{self._rank_id}"]).row)

        return app_mems

    def _combine_ge_ms_memory_record(self) -> list:
        """Combine ge and mindspore memory record data"""
        memory_records = []
        for ge_memory in self._ge_memory_record:
            memory_record = dict(zip(self.GE_MEMORY_RECORD_HEADERS, ge_memory))
            memory_records.append(
                MemoryRecordBean([memory_record.get("Component", "GE"), memory_record.get("Timestamp(us)"),
                                  memory_record.get("Total Allocated(KB)", 0),
                                  memory_record.get("Total Reserved(KB)", 0),
                                  0, memory_record.get("Device")]))
        for ms_memory in self._ms_memory_record:
            memory_record = dict(zip(self.MS_MEMORY_RECORD_HEADERS, ms_memory))
            memory_records.append(
                MemoryRecordBean(["MindSpore", Decimal(memory_record.get("Timestamp(ns)", 0)) / 1000,
                                  float(memory_record.get("Total Allocated(Byte)", 0)) / 1024,
                                  float(memory_record.get("Total Reserved(Byte)", 0)) / 1024,
                                  float(memory_record.get("Total Active(Byte)", 0)) / 1024,
                                  f"NPU:{self._rank_id}"]))
        memory_records.sort(key=lambda x: x.time_us)
        last_ge_memory, last_ms_memory = MemoryRecordBean([0] * 6), MemoryRecordBean([0] * 6)
        result_data = []
        for memory_record in memory_records:
            result_data.append(memory_record.row)
            last_memory = last_ms_memory if memory_record.is_ge_component() else last_ge_memory
            combined_mem = MemoryRecordBean(["MindSpore+GE", memory_record.time_us,
                                             memory_record.total_allocated_kb + last_memory.total_allocated_kb,
                                             memory_record.total_reserved_kb + last_memory.total_reserved_kb,
                                             memory_record.total_active_kb + last_memory.total_active_kb,
                                             f"NPU:{self._rank_id}"])
            result_data.append(combined_mem.row)
            if memory_record.is_ge_component():
                last_ge_memory = memory_record
            else:
                last_ms_memory = memory_record
        return result_data + self._get_app_reserved_memory()


class MemoryRecordBean:
    """Memory Record Struct"""
    KB_TO_MB = 1000
    HEADERS = ["Component", "Timestamp(us)", "Total Allocated(KB)", "Total Reserved(KB)",
               "Total Active(KB)", "Device Type"]

    def __init__(self, data: list):
        self._data = dict(zip(self.HEADERS, data))

    @property
    def row(self) -> list:
        return [self.component, self.time_us_str, self.total_allocated_mb, self.total_reserved_mb,
                self.total_active_mb, self.device_type]

    @property
    def component(self) -> str:
        return self._data.get("Component", "")

    @property
    def time_us_str(self) -> str:
        ts_us = self._data.get("Timestamp(us)", 0)
        return str(ts_us)

    @property
    def total_allocated_mb(self) -> float:
        return float(self._data.get("Total Allocated(KB)", 0)) / self.KB_TO_MB

    @property
    def total_reserved_mb(self) -> float:
        return float(self._data.get("Total Reserved(KB)", 0)) / self.KB_TO_MB

    @property
    def total_active_mb(self) -> float:
        return float(self._data.get("Total Active(KB)", 0)) / self.KB_TO_MB

    @property
    def device_type(self) -> float:
        return self._data.get("Device Type", "")

    @property
    def total_allocated_kb(self) -> float:
        return float(self._data.get("Total Allocated(KB)", 0))

    @property
    def total_reserved_kb(self) -> float:
        return float(self._data.get("Total Reserved(KB)", 0))

    @property
    def total_active_kb(self) -> float:
        return float(self._data.get("Total Active(KB)", 0))

    @property
    def time_us(self) -> Decimal:
        return Decimal(self._data.get("Timestamp(us)", 0))

    def is_ge_component(self):
        """
        Determine if it is GE
        """
        return self.component == "GE"
