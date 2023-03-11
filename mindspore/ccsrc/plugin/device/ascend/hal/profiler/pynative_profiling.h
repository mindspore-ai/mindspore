/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PYNATIVE_PROFILING_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PYNATIVE_PROFILING_H_

#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace profiler {
namespace ascend {
using mindspore::device::ascend::AscendKernelRuntime;

struct PynativeOpInfo {
  std::string op_name;
  int thread_index;
  // the unit is ms
  double_t start_timestamp = 0l;
  // the unit is ms
  double_t duration = 0l;
  void *stream{nullptr};
  std::shared_ptr<DeviceEvent> start;
  std::shared_ptr<DeviceEvent> end;
};

class MS_CORE_API PynativeProfiler : public Profiler {
 public:
  static std::shared_ptr<PynativeProfiler> GetInstance();
  PynativeProfiler() = default;
  ~PynativeProfiler() {}
  void Init(const std::string &profiling_path, uint32_t device_id, const std::string &profiling_options) override;
  void Finalize() override {}
  void Start() override {}
  void Stop() override;
  void OpDataProducerBegin(AscendKernelRuntime *runtime_instance_, void *stream, std::thread::id thread_id,
                           const std::string &op_name, bool is_dynamic_shape);
  void OpDataProducerEnd() override;
  void OpDataProducerEnd(std::thread::id thread_id, bool is_dynamic_shape);
  void StepProfilingEnable(const bool enable_flag) override;

 private:
  void WriteOpDetail(const std::string &out_path_dir);
  void WriteStartTime();
  void SaveProfileData() override;
  void ClearInst() override;
  int NewThreadIndex();

  std::int32_t rank_id_;
  std::vector<PynativeOpInfo> pynative_op_info_;
  bool enable_flag_ = false;
  const uint64_t kUSecondInSecond = 1000000;
  std::map<std::thread::id, PynativeOpInfo> thread_op_info_map_;
  std::shared_mutex op_map_mutex_;
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PYNATIVE_PROFILING_H_
