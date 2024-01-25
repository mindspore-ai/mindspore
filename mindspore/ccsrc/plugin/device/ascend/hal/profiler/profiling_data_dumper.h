/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <unistd.h>
#include <sys/stat.h>
#include <linux/limits.h>
#include <libgen.h>
#include <fcntl.h>
#include <sys/syscall.h>
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

namespace mindspore {
namespace profiler {
namespace ascend {
constexpr uint32_t kDefaultRingBuffer = 1024;
constexpr uint32_t kBatchMaxLen = 5 * 1024 * 1024;  // 5 MB
constexpr uint32_t kMaxWaitTimeUs = 1024;
constexpr uint32_t kNotifyInterval = 2;

class Utils {
 public:
  static bool IsFileExist(const std::string &path);
  static bool IsFileWritable(const std::string &path);
  static bool IsDir(const std::string &path);
  static bool CreateDir(const std::string &path);
  static std::string RealPath(const std::string &path);
  static std::string RelativeToAbsPath(const std::string &path);
  static std::string DirName(const std::string &path);
  static uint64_t GetClockMonotonicRawNs();
  static uint64_t GetClockSyscnt();
  static uint64_t GetClockTime();
  static bool CreateFile(const std::string &path);
  static bool IsSoftLink(const std::string &path);
  static uint64_t GetTid();
  static uint64_t GetPid();
};

template <typename T>
class RingBuffer {
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

class Thread {
 public:
  Thread() : is_alive_(false), pid_(0), thread_name_("NPUProfiler") {}

  ~Thread() {
    if (is_alive_) {
      (void)pthread_cancel(pid_);
      (void)pthread_join(pid_, nullptr);
    }
  }

  void SetThreadName(const std::string &name) {
    if (!name.empty()) {
      thread_name_ = name;
    }
  }

  std::string GetThreadName() { return thread_name_; }

  int Start() {
    int ret = pthread_create(&pid_, nullptr, Execute, reinterpret_cast<void *>(this));
    is_alive_ = (ret == 0) ? true : false;
    return ret;
  }

  int Stop() { return Join(); }

  int Join() {
    int ret = pthread_join(pid_, nullptr);
    is_alive_ = (ret == 0) ? false : true;
    return ret;
  }

 private:
  static void *Execute(void *args) {
    Thread *thr = reinterpret_cast<Thread *>(args);
    thr->Run();
    return nullptr;
  }
  virtual void Run() = 0;

 private:
  bool is_alive_;
  pthread_t pid_;
  std::string thread_name_;
};

struct BaseReportData {
  int32_t device_id{0};
  std::string tag;
  BaseReportData(int32_t device_id, std::string tag) : device_id(device_id), tag(std::move(tag)) {}
  virtual ~BaseReportData() = default;
  virtual std::vector<uint8_t> encode() = 0;
};

class ProfilingDataDumper : public Thread {
 public:
  ProfilingDataDumper();
  virtual ~ProfilingDataDumper();
  void Init(const std::string &path, size_t capacity = kDefaultRingBuffer);
  void UnInit();
  void Report(std::unique_ptr<BaseReportData> data);
  void Start();
  void Stop();

  static std::shared_ptr<ProfilingDataDumper> &GetInstance() {
    static std::shared_ptr<ProfilingDataDumper> instance = std::make_shared<ProfilingDataDumper>();
    return instance;
  }

 private:
  void Flush();
  void Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap);
  void Run();
  void GatherAndDumpData();

 private:
  std::string path_;
  std::atomic<bool> start_;
  std::atomic<bool> init_;
  RingBuffer<std::unique_ptr<BaseReportData>> data_chunk_buf_;
  std::map<std::string, FILE *> fd_map_;
};

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_DATA_DUMPER_H_
