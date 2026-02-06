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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_UTILS_H
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_UTILS_H
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <dlfcn.h>
#include <fcntl.h>
#include <libgen.h>
#include <linux/limits.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <vector>
#include <string>
#include <cstring>
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "utils/distributed_meta.h"

constexpr int EXPECT_FALSE = 0;
constexpr int EXPECT_TRUE = 1;

#define UNLIKELY(x) __builtin_expect(!!(x), EXPECT_FALSE)
#define LIKELY(x) __builtin_expect(!!(x), EXPECT_TRUE)

namespace mindspore {
namespace profiler {

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
class Utils {
 private:
  static constexpr mode_t DEFAULT_DIR_MODE = 0750;
  static constexpr uint64_t NANOSECONDS_PER_SECOND = 1000000000ULL;

 public:
  static bool IsFileExist(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
  }

  static bool IsFileWritable(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return (access(path.c_str(), W_OK) == 0) ? true : false;
  }

  static bool IsDir(const std::string &path) {
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

  static bool CreateDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    if (IsFileExist(path)) {
      return IsDir(path) ? true : false;
    }
    size_t pos = 0;
    while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
      std::string base_dir = path.substr(0, ++pos);
      if (IsFileExist(base_dir)) {
        if (IsDir(base_dir)) {
          continue;
        } else {
          return false;
        }
      }
      if (mkdir(base_dir.c_str(), DEFAULT_DIR_MODE) != 0) {
        return false;
      }
    }
    return (mkdir(path.c_str(), DEFAULT_DIR_MODE) == 0) ? true : false;
  }

  static std::string RealPath(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return "";
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPath) == nullptr) {
      return "";
    }
    return std::string(realPath);
  }

  static std::string RelativeToAbsPath(const std::string &path) {
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

  static std::string DirName(const std::string &path) {
    if (path.empty()) {
      return "";
    }
    std::vector<char> temp_path(path.begin(), path.end());
    temp_path.push_back('\0');
    char *path_c = dirname(temp_path.data());
    return path_c ? std::string(path_c) : "";
  }

  static uint64_t GetClockMonotonicRawNs() {
    struct timespec ts = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * NANOSECONDS_PER_SECOND + static_cast<uint64_t>(ts.tv_nsec);
  }

  static uint64_t getClockSyscnt() {
    uint64_t cycles;
#if defined(__aarch64__)
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
#elif defined(__x86_64__)
    constexpr uint32_t uint32Bits = 32U;
    uint32_t hi = 0;
    uint32_t lo = 0;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    cycles = (static_cast<uint64_t>(lo)) | ((static_cast<uint64_t>(hi)) << uint32Bits);
#elif defined(__arm__)
    const uint32_t uint32Bits = 32U;
    uint32_t hi = 0;
    uint32_t lo = 0;
    asm volatile("mrrc p15, 1, %0, %1, c14" : "=r"(lo), "=r"(hi));
    cycles = (static_cast<uint64_t>(lo)) | ((static_cast<uint64_t>(hi)) << uint32Bits);
#else
    cycles = 0;
#endif
    return cycles;
  }

  static bool CreateFile(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX || !CreateDir(DirName(path))) {
      return false;
    }
    int fd = creat(path.c_str(), S_IRUSR | S_IWUSR | S_IRGRP);
    return (fd < 0 || close(fd) != 0) ? false : true;
  }

  static std::optional<std::string> CreatePrefixPath(const std::string &input_path,
                                                     const bool support_relative_path = false) {
    std::optional<std::string> prefix_path;
    std::optional<std::string> file_name;

    FileUtils::SplitDirAndFileName(input_path, &prefix_path, &file_name);
    if (!file_name.has_value()) {
      MS_LOG(ERROR) << "Cannot get file_name from: " << input_path;
      return std::nullopt;
    }

    auto file_name_str = file_name.value();

#if defined(SYSTEM_ENV_POSIX)
    if (file_name_str.length() > NAME_MAX) {
      MS_LOG(ERROR) << "The length of file name: " << file_name_str.length() << " exceeds limit: " << NAME_MAX;
      return std::nullopt;
    }
#endif

    std::string prefix_path_str;
    if (prefix_path.has_value()) {
      auto create_prefix_path = FileUtils::CreateNotExistDirs(prefix_path.value(), support_relative_path);
      if (!create_prefix_path.has_value()) {
        return std::nullopt;
      }
      prefix_path_str = create_prefix_path.value();
    } else {
      auto pwd_path = FileUtils::GetRealPath("./");
      if (!pwd_path.has_value()) {
        MS_LOG(ERROR) << "Cannot get pwd path";
        return std::nullopt;
      }
      prefix_path_str = pwd_path.value();
    }

    return std::string(prefix_path_str + "/" + file_name_str);
  }

  static std::string GetSaveGraphsPathName(const std::string &file_name, const std::string &save_path = "") {
    std::string save_graphs_path;
    if (save_path.empty()) {
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      save_graphs_path = ms_context->GetSaveGraphsPath();
      if (save_graphs_path.empty()) {
        save_graphs_path = ".";
      }
    } else {
      save_graphs_path = save_path;
    }
    uint32_t rank_id = DistributedMeta::GetInstance()->global_rank_id();

    return save_graphs_path + "/rank_" + std::to_string(rank_id) + "/" + file_name;
  }

  static void *GetLibHandler(const std::string &lib_path, bool if_global = false) {
    void *handler = nullptr;
    auto flag = if_global ? RTLD_GLOBAL : RTLD_LAZY;
    handler = dlopen(lib_path.c_str(), flag);
    if (handler == nullptr) {
      MS_LOG(ERROR) << "Dlopen " << lib_path << " failed! " << dlerror();
    }
    return handler;
  }
};
#else
class Utils {
 public:
  static bool IsFileExist(const std::string &path) { return true; }
  static bool IsFileWritable(const std::string &path) { return true; }
  static bool IsDir(const std::string &path) { return true; }
  static bool CreateDir(const std::string &path) { return true; }
  static std::string RealPath(const std::string &path) { return ""; }
  static std::string RelativeToAbsPath(const std::string &path) { return ""; }
  static std::string DirName(const std::string &path) { return ""; }
  static uint64_t GetClockMonotonicRawNs() { return 0; }
  static uint64_t getClockSyscnt() { return 0; }
  static bool CreateFile(const std::string &path) { return true; }
  static std::string GetSaveGraphsPathName(const std::string &file_name, const std::string &save_path = "") {
    return "";
  }
  static std::optional<std::string> CreatePrefixPath(const std::string &input_path,
                                                     const bool support_relative_path = false) {
    return "";
  }
  static void *GetLibHandler(const std::string &lib_path, bool if_global = false) { return nullptr; }
};
#endif
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_UTILS_H
