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

#ifndef MINDSPORE_RT_CALLBACK_MANAGER_H_
#define MINDSPORE_RT_CALLBACK_MANAGER_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <utility>
#include "profiler/device/ascend/blocking_queue.h"
#include "runtime/base.h"

namespace mindspore {
namespace profiler {
namespace ascend {
using rtCallback_t = std::function<void(void *)>;
enum Status { kSuccess = 0, kFail, kInvalidParam };
class CallbackManager {
 public:
  static CallbackManager &GetInstance(rtStream_t stream) {
    static CallbackManager instance(stream);
    return instance;
  }

  explicit CallbackManager(rtStream_t stream);

  ~CallbackManager() = default;

  Status Init();

  Status Destroy();

  Status RegisterCallback(rtCallback_t callback, void *user_data);
  Status RegisterCallback(const std::function<void()> &callback);

 private:
  Status CallbackProcess();
  static void RtCallbackFunc(void *data);

  BlockingQueue<std::pair<rtEvent_t, std::pair<rtCallback_t, void *>>> callback_queue_;
  rtStream_t stream_;
  std::future<Status> ret_future_;
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_RT_CALLBACK_MANAGER_H_
