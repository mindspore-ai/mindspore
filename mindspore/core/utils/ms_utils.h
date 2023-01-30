/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_UTILS_MS_UTILS_H_
#define MINDSPORE_CORE_UTILS_MS_UTILS_H_

#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <limits>
#include <cmath>
#include <chrono>
#include "mindapi/base/macros.h"
namespace mindspore {
class MSLogTime {
 public:
  MSLogTime() {}
  ~MSLogTime() {}
  inline void Start() { this->start = std::chrono::system_clock::now(); }
  inline void End() { this->end = std::chrono::system_clock::now(); }
  uint64_t GetRunTimeUS() {
    auto ms_duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end - this->start);
    uint64_t ms = ms_duration.count();
    return ms;
  }

 private:
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
};
}  // namespace mindspore

#define DISABLE_COPY_AND_ASSIGN(ClassType) \
  ClassType(const ClassType &) = delete;   \
  ClassType &operator=(const ClassType &) = delete;

#define TRY_AND_CATCH_WITH_EXCEPTION(expr, error_msg)                               \
  do {                                                                              \
    try {                                                                           \
      (expr);                                                                       \
    } catch (const std::exception &e) {                                             \
      MS_LOG(EXCEPTION) << "Caught exception of " << e.what() << ". " << error_msg; \
    }                                                                               \
  } while (0)

namespace mindspore {
namespace common {
inline const char *SafeCStr(const std::string &str) { return str.c_str(); }
MS_CORE_API const char *SafeCStr(const std::string &&str);

static inline std::string GetEnv(const std::string &envvar) {
  const char *value = std::getenv(envvar.c_str());

  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

static inline int SetEnv(const char *envname, const char *envvar, int overwrite = 1) {
#if defined(_WIN32)
  return 0;
#else
  return ::setenv(envname, envvar, overwrite);
#endif
}

static inline void SetOMPThreadNum() {
  const size_t kOMPThreadMaxNum = 16;
  const size_t kOMPThreadMinNum = 1;
  // The actor concurrent execution max num.
  const size_t kActorConcurrentMaxNum = 4;

  size_t cpu_core_num = std::thread::hardware_concurrency();
  size_t cpu_core_num_half = cpu_core_num / 2;
  // Ensure that the calculated number of OMP threads is at most half the number of CPU cores.
  size_t OMP_thread_num = cpu_core_num_half / kActorConcurrentMaxNum;

  OMP_thread_num = OMP_thread_num < kOMPThreadMinNum ? kOMPThreadMinNum : OMP_thread_num;
  OMP_thread_num = OMP_thread_num > kOMPThreadMaxNum ? kOMPThreadMaxNum : OMP_thread_num;

  std::string OMP_env = std::to_string(OMP_thread_num);
  (void)SetEnv("OMP_NUM_THREADS", OMP_env.c_str(), 0);
}

static inline bool IsLittleByteOrder() {
  uint32_t check_code = 0x12345678;
  auto check_pointer = reinterpret_cast<uint8_t *>(&check_code);
  uint8_t head_code = 0x78;
  if (check_pointer[0] == head_code) {
    return true;
  }
  return false;
}

static inline bool UseMPI() {
  // If these OpenMPI environment variables are set, we consider this process is launched by OpenMPI.
  std::string ompi_command_env = GetEnv("OMPI_COMMAND");
  std::string pmix_rank_env = GetEnv("PMIX_RANK");
  if (!ompi_command_env.empty() && !pmix_rank_env.empty()) {
    return true;
  }
  return false;
}

static inline bool UseDynamicCluster() {
  // If environment variable 'MS_ROLE' is set, we consider this process is participating in cluster building.
  return !common::GetEnv("MS_ROLE").empty();
}

// UseDynamicCluster or UseMPI. If false, means use rank table file.
static inline bool UseHostCollective() { return common::UseDynamicCluster() || common::UseMPI(); }

template <typename T>
bool IsEqual(const T *a, const T *b) {
  if (a == b) {
    return true;
  }
  if (a == nullptr || b == nullptr) {
    return false;
  }
  return *a == *b;
}

template <typename T>
bool IsEqual(const std::shared_ptr<T> &a, const std::shared_ptr<T> &b) {
  return IsEqual(a.get(), b.get());
}

template <typename T>
bool IsAttrsEqual(const T &a, const T &b) {
  if (&a == &b) {
    return true;
  }
  if (a.size() != b.size()) {
    return false;
  }
  auto iter1 = a.begin();
  auto iter2 = b.begin();
  while (iter1 != a.end()) {
    if (iter1->first != iter2->first) {
      return false;
    }
    if (!IsEqual(iter1->second, iter2->second)) {
      return false;
    }
    ++iter1;
    ++iter2;
  }
  return true;
}

inline bool IsFloatEqual(const float &a, const float &b) {
  return (std::fabs(a - b) <= std::numeric_limits<float>::epsilon());
}

inline bool IsDoubleEqual(const double &a, const double &b) {
  return (std::fabs(a - b) <= std::numeric_limits<double>::epsilon());
}
}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_UTILS_H_
