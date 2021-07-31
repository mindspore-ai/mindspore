/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#define DISABLE_COPY_AND_ASSIGN(ClassType) \
  ClassType(const ClassType &) = delete;   \
  ClassType &operator=(const ClassType &) = delete;

namespace mindspore {
namespace common {
inline const char *SafeCStr(const std::string &str) { return str.c_str(); }
const char *SafeCStr(const std::string &&str);

static inline std::string GetEnv(const std::string &envvar) {
  const char *value = ::getenv(envvar.c_str());

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
  size_t cpu_core_num = std::thread::hardware_concurrency();
  size_t cpu_core_num_half = cpu_core_num / 2;
  const size_t kOMPThreadMaxNum = 16;
  const size_t kOMPThreadMinNum = 1;

  size_t OMP_thread_num = cpu_core_num_half < kOMPThreadMinNum ? kOMPThreadMinNum : cpu_core_num_half;
  OMP_thread_num = OMP_thread_num > kOMPThreadMaxNum ? kOMPThreadMaxNum : OMP_thread_num;

  std::string OMP_env = std::to_string(OMP_thread_num);
  (void)SetEnv("OMP_NUM_THREADS", OMP_env.c_str(), 0);
}
}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_UTILS_H_
