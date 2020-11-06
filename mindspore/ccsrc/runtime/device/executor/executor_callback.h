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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_EXECUTOR_EXECUTOR_CALLBACK_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_EXECUTOR_EXECUTOR_CALLBACK_H_

#include <queue>
#include <mutex>
#include <functional>
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
class ExecutorCallback {
 public:
  static ExecutorCallback &GetInstance() {
    static ExecutorCallback instance;
    return instance;
  }

  void RegistCallback(const std::function<void()> &callback);
  void Consume();

 private:
  ExecutorCallback() = default;
  ~ExecutorCallback() = default;
  DISABLE_COPY_AND_ASSIGN(ExecutorCallback);

  std::queue<std::function<void()>> callback_queue_;
  std::mutex lock_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_EXECUTOR_EXECUTOR_CALLBACK_H_
