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

#ifndef MINDSPORE_CCSRC_PROFILER_DEVICE_CPU_CPU_DATA_SAVER_H
#define MINDSPORE_CCSRC_PROFILER_DEVICE_CPU_CPU_DATA_SAVER_H
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include "profiler/device/cpu/cpu_profiling.h"
#include "profiler/device/data_saver.h"
namespace mindspore {
namespace profiler {
namespace cpu {
class CpuDataSaver : public DataSaver {
 public:
  CpuDataSaver() = default;

  ~CpuDataSaver() = default;

  CpuDataSaver(const CpuDataSaver &) = delete;

  CpuDataSaver &operator=(const CpuDataSaver &) = delete;

  void WriteFile(std::string out_path);
};
}  // namespace cpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PROFILER_DEVICE_CPU_CPU_DATA_SAVER_H
