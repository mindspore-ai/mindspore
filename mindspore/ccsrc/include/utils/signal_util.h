/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SIGNAL_UTIL_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SIGNAL_UTIL_H_

#include <csignal>
#include "include/utils/visible.h"

namespace mindspore {
typedef void (*IntHandlerFunc)(int, siginfo_t *, void *);
class COMMON_EXPORT SignalGuard {
 public:
  explicit SignalGuard(IntHandlerFunc func);
  ~SignalGuard();

 private:
  void RegisterHandlers(IntHandlerFunc func);
  void (*old_handler)(int, siginfo_t *, void *) = nullptr;
  struct sigaction int_action;
};
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
COMMON_EXPORT bool RegisterGlobalSignalHandler(IntHandlerFunc handler);
COMMON_EXPORT void DefaultIntHandler(int, siginfo_t *, void *);
#endif
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SIGNAL_UTIL_H_
