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
#include "include/fork_utils.h"

namespace mindspore {
ForkUtils &ForkUtils::GetInstance() noexcept {
  static ForkUtils instance;
  return instance;
}

// Function called in parent process before fork.
void ForkUtils::BeforeFork() {
  FORK_UTILS_LOG("Fork event occurred, function called in parent process before fork.");
  for (auto &iter : ForkUtils::GetInstance().GetCallbacks()) {
    iter.before_fork_func();
  }
}

// Function called in parent process after fork.
void ForkUtils::ParentAtFork() {
  FORK_UTILS_LOG("Fork event occurred, function called in parent process after fork.");
  for (auto &iter : ForkUtils::GetInstance().GetCallbacks()) {
    iter.parent_atfork_func();
  }
}

// Function called in child process after fork.
void ForkUtils::ChildAtFork() {
  FORK_UTILS_LOG("Fork event occurred, function called in child process after fork.");
  for (auto &iter : ForkUtils::GetInstance().GetCallbacks()) {
    iter.child_atfork_func();
  }
}

// Function called when fork callback function is nullptr.
void EmptyFunction() {}
}  // namespace mindspore
