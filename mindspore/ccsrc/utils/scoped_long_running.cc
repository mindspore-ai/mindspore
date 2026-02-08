/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/utils/scoped_long_running.h"

namespace mindspore {
ScopedLongRunning::ScopedLongRunning() {
  if (hook_ != nullptr) {
    hook_->Enter();
  }
}

ScopedLongRunning::~ScopedLongRunning() {
  if (hook_ != nullptr) {
    hook_->Leave();
  }
}

void ScopedLongRunning::SetHook(ScopedLongRunningHookPtr hook) {
  if (hook_ == nullptr) {
    hook_ = std::move(hook);
  }
}
}  // namespace mindspore
