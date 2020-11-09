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

#include "runtime/device/executor/executor_callback.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
void ExecutorCallback::RegistCallback(const std::function<void()> &callback) {
  std::lock_guard<std::mutex> guard(lock_);
  callback_queue_.push(callback);
}

void ExecutorCallback::Consume() {
  std::lock_guard<std::mutex> guard(lock_);
  while (!callback_queue_.empty()) {
    auto callback_func = callback_queue_.front();
    callback_queue_.pop();
    if (!callback_func) {
      MS_LOG(EXCEPTION) << "callback_func is empty";
    }
    callback_func();
  }
}
}  // namespace device
}  // namespace mindspore
