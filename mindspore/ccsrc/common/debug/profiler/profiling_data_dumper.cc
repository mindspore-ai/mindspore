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
#include "common/debug/profiler/profiling_data_dumper.h"
#include <algorithm>
#include <mutex>
#include <utility>
#include "kernel/kernel.h"
#include "ops/structure_op_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "nlohmann/json.hpp"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace profiler {
namespace ascend {

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
bool Utils::IsFileExist(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return false;
  }
  return (access(path.c_str(), F_OK) == 0) ? true : false;
}

bool Utils::IsFileWritable(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return false;
  }
  return (access(path.c_str(), W_OK) == 0) ? true : false;
}

bool Utils::IsDir(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return false;
  }
  struct stat st = {0};
  int ret = lstat(path.c_str(), &st);
  if (ret != 0) {
    return false;
  }
  return S_ISDIR(st.st_mode) ? true : false;
}

bool Utils::CreateDir(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return false;
  }
  if (IsFileExist(path)) {
    return IsDir(path) ? true : false;
  }
  size_t pos = 0;
  static const int DEFAULT_MKDIR_MODE = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
    std::string base_dir = path.substr(0, ++pos);
    if (IsFileExist(base_dir)) {
      if (IsDir(base_dir)) {
        continue;
      } else {
        return false;
      }
    }
    if (mkdir(base_dir.c_str(), DEFAULT_MKDIR_MODE) != 0) {
      return false;
    }
  }
  return (mkdir(path.c_str(), DEFAULT_MKDIR_MODE) == 0) ? true : false;
}

std::string Utils::RealPath(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return "";
  }
  char realPath[PATH_MAX] = {0};
  if (realpath(path.c_str(), realPath) == nullptr) {
    return "";
  }
  return std::string(realPath);
}

std::string Utils::RelativeToAbsPath(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return "";
  }
  if (path[0] != '/') {
    char pwd_path[PATH_MAX] = {0};
    if (getcwd(pwd_path, PATH_MAX) != nullptr) {
      return std::string(pwd_path) + "/" + path;
    }
    return "";
  }
  return std::string(path);
}

std::string Utils::DirName(const std::string &path) {
  if (path.empty()) {
    return "";
  }
  std::string temp_path = std::string(path.begin(), path.end());
  char *path_c = dirname(const_cast<char *>(temp_path.data()));
  return path_c ? std::string(path_c) : "";
}

uint64_t Utils::GetClockMonotonicRawNs() {
  struct timespec ts = {0};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000 +
         static_cast<uint64_t>(ts.tv_nsec);  // To convert to nanoseconds, it needs to be 1000000000.
}

bool Utils::CreateDumpFile(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX || !CreateDir(DirName(path))) {
    return false;
  }
  std::ofstream output_file(path);
  output_file.close();
  if (chmod(path.c_str(), S_IRUSR | S_IWUSR | S_IRGRP) == -1) {
    MS_LOG(WARNING) << "chmod failed, path: " << path;
    return false;
  }
  return true;
}

bool Utils::IsSoftLink(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX || !IsFileExist(path)) {
    return false;
  }
  struct stat st {};
  if (lstat(path.c_str(), &st) != 0) {
    return false;
  }
  return S_ISLNK(st.st_mode);
}

uint64_t Utils::GetTid() {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  static thread_local uint64_t tid = static_cast<uint64_t>(syscall(SYS_gettid));
#else
  static thread_local uint64_t tid = 0;
#endif
  return tid;
}

uint64_t Utils::GetPid() {
  static thread_local uint64_t pid = static_cast<uint64_t>(getpid());
  return pid;
}

template <typename T>
void RingBuffer<T>::Init(size_t capacity) {
  capacity_ = capacity;
  data_queue_.resize(capacity);
  is_inited_ = true;
  is_quit_ = false;
}

template <typename T>
void RingBuffer<T>::UnInit() {
  if (is_inited_) {
    data_queue_.clear();
    read_index_ = 0;
    write_index_ = 0;
    idle_write_index_ = 0;
    capacity_ = 0;
    is_quit_ = true;
    is_inited_ = false;
  }
}

template <typename T>
size_t RingBuffer<T>::Size() {
  size_t curr_read_index = read_index_.load(std::memory_order_acquire);
  size_t curr_write_index = write_index_.load(std::memory_order_acquire);
  if (curr_read_index >= curr_write_index) {
    return 0;
  }
  return curr_write_index - curr_read_index;
}

template <typename T>
bool RingBuffer<T>::Full() {
  size_t curr_write_index = write_index_.load(std::memory_order_acquire);
  if (curr_write_index >= capacity_) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
bool RingBuffer<T>::Push(T data) {
  size_t curr_write_index = 0;
  curr_write_index = write_index_.fetch_add(1, std::memory_order_acquire);
  if (curr_write_index >= capacity_) {
    return false;
  }
  data_queue_[curr_write_index] = std::move(data);
  return true;
}

template <typename T>
T RingBuffer<T>::Pop() {
  if (!is_inited_) {
    return nullptr;
  }
  size_t curr_read_index = read_index_.fetch_add(1, std::memory_order_acquire);
  size_t curr_write_index = write_index_.load(std::memory_order_acquire);
  if (curr_read_index >= curr_write_index || curr_read_index >= capacity_) {
    return nullptr;
  }
  T data = std::move(data_queue_[curr_read_index]);
  return data;
}

template <typename T>
void RingBuffer<T>::Reset() {
  write_index_ = 0;
  read_index_ = 0;
}

ProfilingDataDumper::ProfilingDataDumper() : path_(""), start_(false), init_(false) {}

ProfilingDataDumper::~ProfilingDataDumper() { UnInit(); }

ProfilingDataDumper &ProfilingDataDumper::GetInstance() {
  static ProfilingDataDumper instance;
  return instance;
}

void ProfilingDataDumper::Init(const std::string &path, size_t capacity) {
  MS_LOG(INFO) << "init profiling data dumper, capacity: " << capacity;
  path_ = path;
  data_chunk_buf_.Init(capacity);
  init_.store(true);
}

void ProfilingDataDumper::UnInit() {
  MS_LOG(INFO) << "uninit profiling data dumper.";
  if (init_.load()) {
    init_.store(false);
    start_.store(false);
    for (auto &f : fd_map_) {
      if (f.second != nullptr) {
        fclose(f.second);
        f.second = nullptr;
      }
    }
    fd_map_.clear();
  }
}

void ProfilingDataDumper::Start() {
  MS_LOG(INFO) << "start profiling data dumper.";
  if (!init_.load() || !Utils::CreateDir(path_)) {
    MS_LOG(ERROR) << "init_.load() " << !init_.load() << !Utils::CreateDir(path_);
    return;
  }
  start_.store(true);
}

void ProfilingDataDumper::Stop() {
  MS_LOG(INFO) << "stop profiling data dumper.";
  if (start_.load() == true) {
    start_.store(false);
  }
  Flush();
}

void ProfilingDataDumper::GatherAndDumpData() {
  std::map<std::string, std::vector<uint8_t>> dataMap;
  uint64_t batchSize = 0;
  while (batchSize < kBatchMaxLen) {
    std::unique_ptr<BaseReportData> data = data_chunk_buf_.Pop();
    if (data == nullptr) {
      break;
    }
    std::vector<uint8_t> encodeData = data->encode();
    batchSize += encodeData.size();
    const std::string &key = data->tag;
    auto iter = dataMap.find(key);
    if (iter == dataMap.end()) {
      dataMap.insert({key, encodeData});
    } else {
      iter->second.insert(iter->second.end(), encodeData.cbegin(), encodeData.cend());
    }
  }
  if (dataMap.size() > 0) {
    Dump(dataMap);
  }
}

void ProfilingDataDumper::Flush() {
  MS_LOG(INFO) << "data_chunk_buf_.Size: " << data_chunk_buf_.Size();
  while (data_chunk_buf_.Size() > 0) {
    GatherAndDumpData();
  }
  data_chunk_buf_.Reset();
}

void ProfilingDataDumper::Report(std::unique_ptr<BaseReportData> data) {
  if (!start_.load() || data == nullptr) {
    return;
  }
  int i = 0;
  while (is_flush_.load() && i < 10) {
    usleep(kMaxWaitTimeUs);
    i++;
  }
  if (!data_chunk_buf_.Push(std::move(data))) {
    is_flush_.store(true);
    std::lock_guard<std::mutex> flush_lock_(flush_mutex_);
    if (data_chunk_buf_.Full()) {
      Flush();
    }
    is_flush_.store(false);
    if (!data_chunk_buf_.Push(std::move(data))) {
      MS_LOG(ERROR) << "profiling data Report failed.";
    }
  }
}

void ProfilingDataDumper::Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap) {
  for (auto &data : dataMap) {
    FILE *fd = nullptr;
    const std::string dump_file = path_ + "/" + data.first;

    auto iter = fd_map_.find(dump_file);
    if (iter == fd_map_.end()) {
      if (!Utils::IsFileExist(dump_file) && !Utils::CreateDumpFile(dump_file)) {
        MS_LOG(WARNING) << "create file failed, dump_file: " << dump_file;
        continue;
      }
      fd = fopen(dump_file.c_str(), "ab");
      if (fd == nullptr) {
        MS_LOG(WARNING) << "create file failed, dump_file: " << dump_file;
        continue;
      }
      fd_map_.insert({dump_file, fd});
    } else {
      fd = iter->second;
    }
    fwrite(reinterpret_cast<const char *>(data.second.data()), sizeof(char), data.second.size(), fd);
    fflush(fd);
  }
}
#else
bool Utils::IsFileExist(const std::string &path) { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

bool Utils::IsFileWritable(const std::string &path) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
bool Utils::IsDir(const std::string &path) { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
bool Utils::CreateDir(const std::string &path) { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

std::string Utils::RealPath(const std::string &path) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

std::string Utils::RelativeToAbsPath(const std::string &path) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

std::string Utils::DirName(const std::string &path) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

uint64_t Utils::GetClockMonotonicRawNs() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

bool Utils::CreateDumpFile(const std::string &path) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

bool Utils::IsSoftLink(const std::string &path) { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

uint64_t Utils::GetTid() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

uint64_t Utils::GetPid() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

template <typename T>
void RingBuffer<T>::Init(size_t capacity) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

template <typename T>
void RingBuffer<T>::UnInit() {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

template <typename T>
size_t RingBuffer<T>::Size() {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

template <typename T>
bool RingBuffer<T>::Full() {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

template <typename T>
bool RingBuffer<T>::Push(T data) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

template <typename T>
T RingBuffer<T>::Pop() {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

template <typename T>
void RingBuffer<T>::Reset() {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

ProfilingDataDumper::ProfilingDataDumper() : path_(""), start_(false), init_(false) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

ProfilingDataDumper::~ProfilingDataDumper() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

ProfilingDataDumper &ProfilingDataDumper::GetInstance() {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

void ProfilingDataDumper::Init(const std::string &path, size_t capacity) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

void ProfilingDataDumper::UnInit() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void ProfilingDataDumper::Start() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void ProfilingDataDumper::Stop() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void ProfilingDataDumper::GatherAndDumpData() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void ProfilingDataDumper::Flush() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }

void ProfilingDataDumper::Report(std::unique_ptr<BaseReportData> data) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}

void ProfilingDataDumper::Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap) {
  MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
}
#endif
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
