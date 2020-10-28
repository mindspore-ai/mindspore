/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATASET_DATASET_PROFILING_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATASET_DATASET_PROFILING_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "profiler/device/gpu/gpu_profiling.h"

using mindspore::profiler::gpu::ProfilingOp;

namespace mindspore {
namespace kernel {
class GetNextProfiling : public ProfilingOp {
 public:
  explicit GetNextProfiling(const std::string &path);
  ~GetNextProfiling() = default;
  void SaveProfilingData();
  void GetDeviceId();
  uint64_t GetTimeStamp() const;
  void RecordData(uint32_t queue_size, uint64_t start_time_stamp, uint64_t end_time_stamp);
  void Init();
  void ChangeFileMode();

 private:
  std::string profiling_path_;
  std::string file_name_;
  std::vector<uint32_t> queue_size_;
  std::vector<std::pair<uint64_t, uint64_t>> time_stamp_;  // First value of std::pair is the start time stamp,
                                                           // Second value of std::pair is the stop time stamp
  std::string device_id_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATASET_DATASET_PROFILING_H_
