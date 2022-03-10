/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/pynative/op_lazy_builder.h"

namespace mindspore::runtime {
void OpLazyBuilder::Register(const std::function<void()> &callback) {
  execute_callback_ = callback;
  registered_ = true;
}

void OpLazyBuilder::Reset() {
  ClearAllResources();
  execute_callback_ = nullptr;
  registered_ = false;
}

void OpLazyBuilder::ClearAllResources() {
  op_build_tasks.clear();
  std::queue<std::shared_ptr<OpTask>> empty;
  std::swap(op_run_tasks, empty);
}

void OpLazyBuilder::ExecuteRemainingTasks() {
  if (!executing_) {
    ExecuteGuard guard;
    if (execute_callback_ != nullptr) {
      execute_callback_();
    }
  }
}
}  // namespace mindspore::runtime
