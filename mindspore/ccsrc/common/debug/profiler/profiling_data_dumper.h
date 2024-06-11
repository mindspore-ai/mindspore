/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include <sys/stat.h>
#include <fcntl.h>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <libgen.h>
#include <linux/limits.h>
#include <sys/syscall.h>
#endif
#include <stdint.h>
#include <fstream>
#include <queue>
#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include "utils/ms_utils.h"
#include "include/common/visible.h"

namespace mindspore {
namespace profiler {
namespace ascend {
constexpr uint32_t kDefaultRingBuffer = 1000 * 1000;
constexpr uint32_t kBatchMaxLen = 5 * 1024 * 1024;  // 5 MB
constexpr uint32_t kMaxWaitTimeUs = 100 * 1000;
constexpr uint32_t kMaxWaitTimes = 10;

class COMMON_EXPORT Utils {
 public:
  static bool IsFileExist(const std::string &path);
  static bool IsFileWritable(const std::string &path);
  static bool IsDir(const std::string &path);
  static bool CreateDir(const std::string &path);
  static std::string RealPath(const std::string &path);
  static std::string RelativeToAbsPath(const std::string &path);
  static std::string DirName(const std::string &path);
  static uint64_t GetClockMonotonicRawNs();
  static bool CreateDumpFile(const std::string &path);
  static bool IsSoftLink(const std::string &path);
  static uint64_t GetTid();
  static uint64_t GetPid();
};

template <typename T>
class COMMON_EXPORT RingBuffer {
 public:
  RingBuffer()
      : is_inited_(false),
        is_quit_(false),
        read_index_(0),
        write_index_(0),
        idle_write_index_(0),
        capacity_(0),
        mask_(0) {}

  ~RingBuffer() { UnInit(); }
  void Init(size_t capacity);
  void UnInit();
  size_t Size();
  bool Push(T data);
  T Pop();
  bool Full();
  void Reset();

 private:
  bool is_inited_;
  volatile bool is_quit_;
  std::atomic<size_t> read_index_;
  std::atomic<size_t> write_index_;
  std::atomic<size_t> idle_write_index_;
  size_t capacity_;
  size_t mask_;
  std::vector<T> data_queue_;
};

struct COMMON_EXPORT BaseReportData {
  int32_t device_id{0};
  std::string tag;
  BaseReportData(int32_t device_id, std::string tag) : device_id(device_id), tag(std::move(tag)) {}
  virtual ~BaseReportData() = default;
  virtual std::vector<uint8_t> encode() = 0;
  virtual void preprocess() = 0;
};

class COMMON_EXPORT ProfilingDataDumper {
 public:
  void Init(const std::string &path, size_t capacity = kDefaultRingBuffer);
  void UnInit();
  void Report(std::unique_ptr<BaseReportData> data);
  void Start();
  void Stop();
  void Flush();

  static ProfilingDataDumper &GetInstance();

 private:
  void Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap);
  void Run();
  void GatherAndDumpData();

 private:
  ProfilingDataDumper();
  virtual ~ProfilingDataDumper();

  std::string path_;
  std::atomic<bool> start_;
  std::atomic<bool> init_;
  std::atomic<bool> is_flush_{false};
  RingBuffer<std::unique_ptr<BaseReportData>> data_chunk_buf_;
  std::map<std::string, FILE *> fd_map_;
  std::mutex flush_mutex_;
  DISABLE_COPY_AND_ASSIGN(ProfilingDataDumper);
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_
