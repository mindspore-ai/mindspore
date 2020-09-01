/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PROFILING_CONTEXT_H_
#define MINDSPORE_PROFILING_CONTEXT_H_

#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cstdint>
#include <unordered_map>
#include "profiler/device/ascend/ascend_profiling.h"
namespace mindspore {
namespace profiler {
namespace ascend {

inline pid_t GetTid() {
  thread_local static pid_t tid = syscall(__NR_gettid);
  return tid;
}

#define RECORD_PROFILING_EVENT(profiler, evt_type, fmt, category, node_name, ...)                                \
  do {                                                                                                           \
    if (profiler != nullptr) {                                                                                   \
      if (node_name != nullptr) {                                                                                \
        profiler->RecordEvent(evt_type, "tid:%lu [%s] [%s] " fmt, GetTid(), node_name, category, ##__VA_ARGS__); \
      } else {                                                                                                   \
        profiler->RecordEvent(evt_type, "tid:%lu [%s] " fmt, GetTid(), category, ##__VA_ARGS__);                 \
      }                                                                                                          \
    }                                                                                                            \
  } while (0)

#define RECORD_MODEL_EXECUTION_EVENT(profiler, fmt, ...) \
  RECORD_PROFILING_EVENT((profiler), kGeneral, fmt, "ModelExecutor", nullptr, ##__VA_ARGS__)

#define RECORD_COMPILE_EVENT(profiler, name, fmt, ...) \
  RECORD_PROFILING_EVENT((profiler), kCompiler, fmt, "Compilation", name, ##__VA_ARGS__)

#define RECORD_EXECUTION_EVENT(profiler, name, fmt, ...) \
  RECORD_PROFILING_EVENT((profiler), kExecution, fmt, "Execution", name, ##__VA_ARGS__)

#define RECORD_CALLBACK_EVENT(profiler, name, fmt, ...) \
  RECORD_PROFILING_EVENT((profiler), kCallback, fmt, "Callback", name, ##__VA_ARGS__)
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_PROFILING_CONTEXT_H_
