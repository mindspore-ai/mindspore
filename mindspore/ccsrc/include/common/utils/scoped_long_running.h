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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SCOPED_LONG_RUNNING_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SCOPED_LONG_RUNNING_H_

#include <memory>
#include <utility>
#include "include/common/visible.h"

namespace mindspore {
// Base Class for scoped long running code.
// Enter() should release some global resource, like Python GIL;
// Leave() should acquire the same global resource released.
class ScopedLongRunningHook {
 public:
  ScopedLongRunningHook() = default;
  virtual ~ScopedLongRunningHook() = default;
  virtual void Enter() = 0;
  virtual void Leave() noexcept = 0;
};
using ScopedLongRunningHookPtr = std::unique_ptr<ScopedLongRunningHook>;

// Before calling into long-running code, construct this RAII class to release global resource
// like Python GIL.
class COMMON_EXPORT ScopedLongRunning {
 public:
  ScopedLongRunning();
  ~ScopedLongRunning();
  static void SetHook(ScopedLongRunningHookPtr hook);

 private:
  inline static ScopedLongRunningHookPtr hook_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SCOPED_LONG_RUNNING_H_
