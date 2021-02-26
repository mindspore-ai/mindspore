/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PROFILER_DEVICE_GPU_GPU_PROFILING_H
#define MINDSPORE_CCSRC_PROFILER_DEVICE_GPU_GPU_PROFILING_H
#include <cuda.h>
#include <cupti.h>
#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "profiler/device/profiling.h"
#include "profiler/device/gpu/gpu_profiling_utils.h"

namespace mindspore {
namespace profiler {
namespace gpu {
enum class CUPTIApiType { kCallback = 0, kActivity = 1 };
enum class ActivityType {
  kKernel = 0,
  kMemcpyH2D = 1,
  kMemcpyD2H = 2,
  kMemcpyH2A = 3,
  kMemcpyA2H = 4,
  kMemcpyA2D = 5,
  kMemcpyD2A = 6,
  kMemcpyD2D = 7,
  kMemcpyP2P = 8,
  kMemcpyH2H = 9,
  kMemset = 10,
  kMemcpyUnknown = 11
};

struct MemcpyInfo {
  size_t bytes;
  unsigned char src_kind;
  unsigned char dst_kind;
};

struct KernelInfo {
  uint64_t registers_per_thread;
  uint64_t static_shared_memory;
  uint64_t dynamic_shared_memory;
  uint64_t block_x;
  uint64_t block_y;
  uint64_t block_z;
  uint64_t grid_x;
  uint64_t grid_y;
  uint64_t grid_z;
};

struct Event {
  std::string kernel_name;
  std::string kernel_type;
  CUPTIApiType api_type;
  ActivityType activity_type;
  uint64_t start_time_stamp;
  uint64_t end_time_stamp;
  std::string op_name;
  uint32_t device_id;
  uint32_t correlation_id;
  uint32_t thread_id;
  uint32_t context_id;
  uint32_t stream_id;
  CUpti_CallbackId cb_id;
  union {
    MemcpyInfo memcpy_info;
    KernelInfo kernel_info;
  };
};

struct BaseTime {
  // nanosecond
  uint64_t host_start_time = 0l;
  uint64_t host_start_monotonic_raw_time = 0l;
  uint64_t gpu_start_time = 0l;
};

const float kTimeUnit = 1000;

class ProfilingOp {
 public:
  ProfilingOp() = default;
  virtual ~ProfilingOp() = default;
  virtual void SaveProfilingData() = 0;
  virtual void Init() = 0;
  std::string Name() const { return op_name_; }

 protected:
  std::string op_name_;
};

class GPUProfiler : public Profiler {
 public:
  static std::shared_ptr<GPUProfiler> GetInstance();
  ~GPUProfiler() { StopCUPTI(); }
  GPUProfiler(const GPUProfiler &) = delete;
  GPUProfiler &operator=(const GPUProfiler &) = delete;

  void Init(const std::string &profileDataPath) override;
  void Stop() override;
  void StopCUPTI();
  void StepProfilingEnable(const bool enable_flag) override;
  void SyncEnable(const bool enable_flag);
  bool GetEnableFlag() const { return enable_flag_; }
  bool GetSyncEnableFlag() const { return sync_enable_flag_; }
  void EventHandleProcess(CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata, const std::string &typestring,
                          uint64_t startTimestamp, uint64_t endTimestamp);
  void CUPTIAPI AllocBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
  void CUPTIAPI ProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
  void OpDataProducerBegin(const std::string op_name, void *stream);
  void OpDataProducerEnd() override;
  void ProcessEvents();
  void RegisterProfilingOp(std::shared_ptr<ProfilingOp> node);
  void SetStepTraceOpName(ProfilingTraceInfo trace_op_name);
  std::string ProfileDataPath() const { return profile_data_path_; }

 private:
  GPUProfiler() = default;
  void OpsParser();
  void EventLog(const Event &event);
  void ClearInst() override;
  void HandleActivityRecord(CUpti_Activity *record);
  void AddEvent(Event &&event);
  void SetRunTimeData(const std::string &op_name, void *stream);
  void FixOpNameByCorrelationId(Event *event);

  static std::shared_ptr<GPUProfiler> profiler_inst_;
  bool enable_flag_ = false;
  bool sync_enable_flag_ = true;
  std::unordered_map<uint32_t, std::string> op_name_map_;
  std::vector<Event> events_;
  BaseTime base_time_;
  std::string op_name_;
  void *stream_;
  void SaveProfileData() override;
  void SaveExtraProfileData();
  std::mutex event_mutex_;

  std::vector<CUpti_ActivityKind> activities_enable_;

  uint64_t cupti_callback_events_count_ = 0l;
  uint64_t cupti_callback_events_drop_count_ = 0l;
  uint64_t max_cupti_callback_events_ = 2 * 1024 * 10000;

  uint64_t cupti_activity_events_count_ = 0l;
  uint64_t cupti_activity_events_drop_count_ = 0l;
  uint64_t max_cupti_activity_events_ = 2 * 1024 * 10000;

  CUpti_SubscriberHandle subscriber_ = nullptr;
  cudaEvent_t op_event_start_;
  cudaEvent_t op_event_stop_;
  uint64_t op_host_time_start_;
  uint64_t op_host_time_stop_;
  uint64_t op_cupti_time_start_;
  std::string profile_data_path_;
  std::map<std::string, std::shared_ptr<ProfilingOp>> profiling_op_;
  ProfilingTraceInfo step_trace_op_name;
};
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PROFILER_DEVICE_GPU_PROFILING_H
