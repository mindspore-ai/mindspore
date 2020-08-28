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

#include "profiler/device/ascend/rt_callback_manager.h"
#include "utils/log_adapter.h"
#include "runtime/event.h"

namespace mindspore {
namespace profiler {
namespace ascend {

CallbackManager::CallbackManager(rtStream_t stream) : stream_(stream) {}

Status CallbackManager::Init() {
  MS_LOG(INFO) << "CallbackManager init, Start to async process event";
  ret_future_ = std::async([&] { return CallbackProcess(); });
  if (!ret_future_.valid()) {
    MS_LOG(ERROR) << "Failed to init callback manager.";
    return kFail;
  }

  return kSuccess;
}

Status CallbackManager::CallbackProcess() {
  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> entry;
  while (true) {
    if (!callback_queue_.Pop(&entry)) {
      MS_LOG(INFO) << "CallbackManager stopped";
      return kFail;
    }

    auto event = entry.first;
    if (event == nullptr) {
      return kSuccess;
    }

    auto rt_err = rtEventSynchronize(event);
    if (rt_err != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtEventSynchronize failed. ret:" << rt_err;
      auto ret = rtEventDestroy(event);
      if (ret != RT_ERROR_NONE) {
        MS_LOG(ERROR) << "rtEventDestroy failed";
      }
      return kFail;
    }

    auto ret = rtEventDestroy(event);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtEventDestroy failed";
    }

    auto cb_func = entry.second.first;
    auto cb_args = entry.second.second;
    cb_func(cb_args);
  }
}

Status CallbackManager::Destroy() {
  MS_LOG(INFO) << "To destroy callback manager.";
  if (!ret_future_.valid()) {
    MS_LOG(INFO) << "CallbackManager not initialized.";
    return kSuccess;
  }

  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> eof_entry;
  eof_entry.first = nullptr;
  callback_queue_.Push(eof_entry);

  auto ret = ret_future_.get();
  MS_LOG(INFO) << "Callback manager ended. ret:" << ret;
  return ret;
}

Status CallbackManager::RegisterCallback(rtCallback_t callback, void *user_data) {
  MS_LOG(INFO) << "To register callback";
  rtEvent_t event = nullptr;
  auto ret = rtEventCreate(&event);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Create event failed";
    return kFail;
  }

  ret = rtEventRecord(event, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Record event failed";
    return kFail;
  }
  auto cb = std::pair<rtCallback_t, void *>(callback, user_data);
  auto entry = std::pair<rtEvent_t, std::pair<rtCallback_t, void *>>(event, std::move(cb));
  if (!callback_queue_.Push(entry)) {
    return kFail;
  }

  MS_LOG(INFO) << "Registering callback successfully";
  return kSuccess;
}

void CallbackManager::RtCallbackFunc(void *data) {
  MS_LOG(INFO) << "To invoke callback function";
  auto callback_func = reinterpret_cast<std::function<void()> *>(data);
  (*callback_func)();
  delete callback_func;
}

Status CallbackManager::RegisterCallback(const std::function<void()> &callback) {
  auto func = std::unique_ptr<std::function<void()>>(new (std::nothrow) std::function<void()>(callback));
  if (func == nullptr) {
    MS_LOG(ERROR) << "callback is nullptr";
    return kInvalidParam;
  }
  MS_LOG(INFO) << "Callback registered";
  return RegisterCallback(RtCallbackFunc, func.release());
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
