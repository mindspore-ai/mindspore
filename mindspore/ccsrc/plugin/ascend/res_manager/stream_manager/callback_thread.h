/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_STREAM_MANAGER_CALLBACKE_THREAD_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_STREAM_MANAGER_CALLBACKE_THREAD_H_

#include <pthread.h>
#include <atomic>
#include <memory>

namespace mindspore {
namespace device {
namespace ascend {
void *callback_thread_func(void *data);

// Callback thread for ascend streams.
struct CallbackThread {
  explicit CallbackThread() : flag_(std::make_shared<std::atomic_bool>(true)) {}
  ~CallbackThread() { cancel(); }

  // pthread_cancel may cause bug now, so just set flag to false.
  void cancel() {
    if (flag_->load()) {
      flag_->store(false);
    }
  }

  int create() {
    flag_->store(true);
    return pthread_create(&thread_, nullptr, &callback_thread_func, &flag_);
  }

  pthread_t thread_{0};
  std::shared_ptr<std::atomic_bool> flag_;
};
using CallbackThreadPtr = std::shared_ptr<CallbackThread>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_STREAM_MANAGER_CALLBACKE_THREAD_H_
