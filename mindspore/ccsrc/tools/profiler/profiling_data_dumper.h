/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "include/utils/visible.h"
#include "tools/profiler/utils.h"
#include "tools/profiler/thread.h"
#include "tools/profiler/report_data.h"
#include "tools/profiler/ring_buffer.h"

namespace mindspore {
namespace profiler {
namespace ascend {
constexpr size_t kNotifyInterval = 104857;  // 0.1 * kDefaultRingBuffer
constexpr size_t kDefaultRingBuffer = 1024 * 1024;
constexpr size_t kBatchMaxLen = 32 * 1024 * 1024;  // 32 MB
constexpr size_t kMaxWaitTimeUs = 1000;

class PROFILER_EXPORT ProfilingDataDumper : public Thread {
 public:
  ProfilingDataDumper();
  virtual ~ProfilingDataDumper();
  void Init(const std::string &path, size_t capacity = kDefaultRingBuffer);
  void UnInit();
  void Report(std::unique_ptr<BaseReportData> data);
  void Start();
  void Stop();
  static ProfilingDataDumper &GetInstance();

 private:
  void Flush();
  void Dump(const std::unordered_map<std::string, std::vector<uint8_t>> &dataMap);
  void Run();
  void GatherAndDumpData();

 private:
  std::string name_;
  std::string path_;
  std::atomic<bool> start_;
  std::atomic<bool> init_;
  RingBuffer<std::unique_ptr<BaseReportData>> data_chunk_buf_;
  std::unordered_map<std::string, FILE *> fd_map_;
  uint64_t dump_count_{0};
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_
