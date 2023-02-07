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

#if defined(__ANDROID__) || defined(MS_COMPILE_OHOS)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#include "src/common/utils.h"
#if defined(_MSC_VER) || defined(_WIN32)
#include <windows.h>
#undef ERROR
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/param.h>
#endif

namespace mindspore {
namespace lite {
uint64_t GetTimeUs() {
#ifdef _MSC_VER
  const int sec_to_us = 1000000;
  LARGE_INTEGER cur_time, frequency;
  QueryPerformanceCounter(&cur_time);
  QueryPerformanceFrequency(&frequency);
  uint64_t sec = cur_time.QuadPart / frequency.QuadPart;
  uint64_t usec = (cur_time.QuadPart % frequency.QuadPart) * sec_to_us / frequency.QuadPart;
  return sec * sec_to_us + usec;
#else
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  // USECS_IN_SEC *NSECS_IN_USEC;
  auto ret_val = static_cast<uint64_t>((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return ret_val;
#endif
}

std::string RemoveSubStr(const std::string &from, const std::string &sub_str, RemoveSubStrMode mode) {
  std::string result = from;
  if (from.empty()) {
    MS_LOG(ERROR) << "string is empty";
    return "";
  }
  if (sub_str.length() > from.length()) {
    MS_LOG(ERROR) << "sub_str is longer than from";
    return "";
  }
  if (mode == PREFIX) {
    if (from.substr(0, sub_str.length()) == sub_str) {
      result = from.substr(sub_str.length());
    }
  } else if (mode == SUFFIX) {
    if (from.rfind(sub_str) == from.length() - sub_str.length()) {
      result = from.substr(0, from.length() - sub_str.length());
    }
  } else {
    size_t index;
    while ((index = result.find(sub_str)) != std::string::npos) {
      result = result.erase(index, sub_str.length());
    }
  }

  return result;
}

std::vector<std::string> StrSplit(const std::string &str, const std::string &delim) {
  if (str.empty()) {
    return {};
  }
  auto start = 0U;
  auto end = str.find(delim);
  std::vector<std::string> substrs;
  while (end != std::string::npos) {
    substrs.push_back(str.substr(start, end - start));
    start = end + delim.length();
    end = str.find(delim, start);
  }
  substrs.push_back(str.substr(start, end));
  return substrs;
}

bool ParseShapeStr(const std::string &shape_str, std::vector<int64_t> *shape_ptr) {
  if (shape_ptr == nullptr) {
    return false;
  }
  auto str_dims = lite::StrSplit(shape_str, ",");
  if (str_dims.empty()) {
    MS_LOG_ERROR << "Invalid input shape dim, dims number cannot be 0";
    return false;
  }
  auto &shape = *shape_ptr;
  shape.resize(str_dims.size());
  for (size_t i = 0; i != str_dims.size(); ++i) {
    int32_t dim = 0;
    if (!ConvertStrToInt(str_dims[i], &dim)) {
      MS_LOG_ERROR << "Invalid input shape dim, dim value range or format is invalid: " << str_dims[i];
      return false;
    }
    if (dim <= 0 && dim != -1) {
      MS_LOG_ERROR << "Invalid input shape dim, dim can only be -1 when dim < 0： " << str_dims[i];
      return false;
    }
    shape[i] = dim;
  }
  return true;
}

bool ConvertStrToInt(const std::string &str, int *value) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "Value is nullptr";
    return false;
  }
  if (str.empty()) {
    return false;
  }
  char *ptr = nullptr;
  constexpr int kBase = 10;
  auto int_val = std::strtol(str.c_str(), &ptr, kBase);
  if (ptr != (str.c_str() + str.size())) {
    return false;
  }
  if (int_val > INT32_MAX || int_val < INT32_MIN || errno == ERANGE) {
    MS_LOG(WARNING) << "The range of value is beyond the range of int32";
    return false;
  }
  *value = static_cast<int32_t>(int_val);
  return true;
}

std::vector<std::string> Tokenize(const std::string &src, const std::string &delimiters,
                                  const Option<size_t> &max_token_num) {
  if (max_token_num.IsSome() && max_token_num.Get() == 0) {
    return {};
  }

  if (src.empty()) {
    MS_LOG(ERROR) << "string is empty";
    return {};
  }

  std::vector<std::string> tokens;
  size_t offset = 0;

  while (true) {
    size_t nonDelimiter = src.find_first_not_of(delimiters, offset);
    if (nonDelimiter == std::string::npos) {
      break;
    }
    size_t delimiter = src.find_first_of(delimiters, nonDelimiter);
    if (delimiter == std::string::npos || (max_token_num.IsSome() && tokens.size() == max_token_num.Get() - 1)) {
      tokens.push_back(src.substr(nonDelimiter));
      break;
    }

    tokens.push_back(src.substr(nonDelimiter, delimiter - nonDelimiter));
    offset = delimiter;
  }
  return tokens;
}

#if defined(__ANDROID__) || defined(MS_COMPILE_OHOS)
uint32_t getHwCap(int hwcap_type) {
  uint32_t ret = getauxval(hwcap_type);
  return ret;
}
#endif

bool IsSupportSDot() {
  bool status = false;
#ifdef ENABLE_ARM64
#if defined(__ANDROID__) || defined(MS_COMPILE_OHOS)
  int hwcap_type = 16;
  uint32_t hwcap = getHwCap(hwcap_type);
  if (hwcap & HWCAP_ASIMDDP) {
    MS_LOG(DEBUG) << "Hw cap support SMID Dot Product, hwcap: 0x" << hwcap;
    status = true;
  } else {
    MS_LOG(DEBUG) << "Hw cap NOT support SIMD Dot Product, hwcap: 0x" << hwcap;
    status = false;
  }
#endif
#endif
  return status;
}

size_t GetMaxMallocSize() {
  size_t max_malloc_size = 0;
#if defined(_MSC_VER) || defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  max_malloc_size = static_cast<size_t>(status.ullTotalPhys);
#else
  max_malloc_size = static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) * static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
  return max_malloc_size;
}

int GetCoreNum() {
  int core_num = 1;
#if defined(_MSC_VER) || defined(_WIN32)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  core_num = sysinfo.dwNumberOfProcessors;
#else
  core_num = sysconf(_SC_NPROCESSORS_CONF);
#endif
  return core_num;
}
}  // namespace lite
}  // namespace mindspore
