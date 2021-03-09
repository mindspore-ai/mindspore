/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "profiler/device/cpu/cpu_profiling.h"

#include <time.h>
#include <cxxabi.h>
#include <cmath>
#include "profiler/device/cpu/cpu_data_saver.h"
#include "pybind_api/api_register.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"

namespace mindspore {
namespace profiler {
namespace cpu {
std::shared_ptr<CPUProfiler> CPUProfiler::profiler_inst_ = nullptr;

std::shared_ptr<CPUProfiler> CPUProfiler::GetInstance() {
  if (profiler_inst_ == nullptr) {
    profiler_inst_ = std::shared_ptr<CPUProfiler>(new (std::nothrow) CPUProfiler());
  }
  return profiler_inst_;
}

void CPUProfiler::Init(const std::string &profileDataPath = "") {
  MS_LOG(INFO) << "Initialize CPU Profiling";
  base_time_ = GetHostMonoTimeStamp();
  profile_data_path_ = profileDataPath;
  MS_LOG(INFO) << " Host start time(ns): " << base_time_ << " profile data path: " << profile_data_path_;
}

void CPUProfiler::StepProfilingEnable(const bool enable_flag) {
  MS_LOG(INFO) << "CPU Profiler enable flag: " << enable_flag;
  enable_flag_ = enable_flag;
}

void CPUProfiler::SetRunTimeData(const std::string &op_name, const uint32_t pid) {
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.op_count += 1;
  } else {
    OpInfo op_info;
    op_info.op_name = op_name;
    op_info.pid = pid;
    op_info.op_count = 1;
    op_info_map_[op_name] = op_info;
  }
  op_name_ = op_name;
  pid_ = pid;
}

void CPUProfiler::OpDataProducerBegin(const std::string op_name, const uint32_t pid) {
  op_time_start_ = GetHostMonoTimeStamp();
  op_time_mono_start_ = GetHostMonoTimeStamp();
  SetRunTimeData(op_name, pid);
}

void CPUProfiler::OpDataProducerEnd() {
  float op_time_elapsed = 0;
  op_time_stop_ = GetHostMonoTimeStamp();
  op_time_elapsed = (op_time_stop_ - op_time_start_) / kNanosecondToMillisecond;
  MS_LOG(DEBUG) << "Host Time Elapsed(us)," << op_name_ << "," << op_time_elapsed;
  Profiler::SetRunTimeData(op_name_, op_time_elapsed);
  Profiler::SetRunTimeData(op_name_, op_time_mono_start_, op_time_elapsed);
}

void CPUProfiler::Stop() {
  MS_LOG(INFO) << "Stop CPU Profiling";
  SaveProfileData();
  ClearInst();
}

void CPUProfiler::SaveProfileData() {
  if (profile_data_path_.empty()) {
    MS_LOG(WARNING) << "Profile data path is empty, skip save profile data.";
  } else {
    CpuDataSaver dataSaver;
    dataSaver.ParseOpInfo(op_info_map_);
    dataSaver.WriteFile(profile_data_path_);
  }
}

void CPUProfiler::ClearInst() { op_info_map_.clear(); }

REGISTER_PYBIND_DEFINE(CPUProfiler_, ([](const py::module *m) {
                         (void)py::class_<CPUProfiler, std::shared_ptr<CPUProfiler>>(*m, "CPUProfiler")
                           .def_static("get_instance", &CPUProfiler::GetInstance, "CPUProfiler get_instance.")
                           .def("init", &CPUProfiler::Init, py::arg("profile_data_path"), "init")
                           .def("stop", &CPUProfiler::Stop, "stop")
                           .def("step_profiling_enable", &CPUProfiler::StepProfilingEnable, py::arg("enable_flag"),
                                "enable or disable step profiling");
                       }));
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore
