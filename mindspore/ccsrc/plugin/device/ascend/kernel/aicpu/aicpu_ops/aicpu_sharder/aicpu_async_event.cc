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
#include "aicpu_sharder/aicpu_async_event.h"
#include <string>
#include "common/kernel_log.h"
#include "aicpu_sharder/aicpu_context.h"

namespace aicpu {
AsyncEventManager &AsyncEventManager::GetInstance() {
  static AsyncEventManager async_event_manager;
  return async_event_manager;
}

void AsyncEventManager::Register(const NotifyFunc &notify) { notify_func_ = notify; }

void AsyncEventManager::NotifyWait(void *notify_param, const uint32_t param_len) {
  if (notify_func_ != nullptr) {
    notify_func_(notify_param, param_len);
  }
}

bool AsyncEventManager::GenTaskInfoFromCtx(AsyncTaskInfo *task_info) const {
  if (task_info == nullptr) {
    AICPU_LOGE("AsyncEventManager GenTaskInfoFromCtx failed, task_info is nullptr.");
    return false;
  }
  (void)aicpu::GetTaskAndStreamId(&task_info->task_id, &task_info->stream_id);
  std::string wait_id_value;
  std::string ker_wait_id(aicpu::kContextKeyWaitId);
  auto status = aicpu::GetThreadLocalCtx(ker_wait_id, &wait_id_value);
  if (status != aicpu::AICPU_ERROR_NONE) {
    AICPU_LOGE("GetThreadLocalCtx failed, ret=%d, key=%s.", status, ker_wait_id.c_str());
    return false;
  }
  task_info->wait_id = static_cast<uint32_t>(atoi(wait_id_value.c_str()));
  std::string wait_type_value;
  std::string key_wait_type(aicpu::kContextKeyWaitType);
  status = aicpu::GetThreadLocalCtx(key_wait_type, &wait_type_value);
  if (status != aicpu::AICPU_ERROR_NONE) {
    AICPU_LOGE("GetThreadLocalCtx failed, ret=%d, key=%s.", status, key_wait_type.c_str());
    return false;
  }
  task_info->wait_type = static_cast<uint8_t>(atoi(wait_type_value.c_str()));
  std::string start_tick_value;
  std::string key_start_tick(aicpu::kContextKeyStartTick);
  status = aicpu::GetThreadLocalCtx(key_start_tick, &start_tick_value);
  if (status != aicpu::AICPU_ERROR_NONE) {
    AICPU_LOGE("GetThreadLocalCtx failed, ret=%d, key=%s.", status, key_start_tick.c_str());
    return false;
  }
  task_info->start_tick = static_cast<uint64_t>(atol(start_tick_value.c_str()));
  status = aicpu::GetOpname(aicpu::GetAicpuThreadIndex(), &task_info->op_name);
  if (status != aicpu::AICPU_ERROR_NONE) {
    AICPU_LOGE("GetOpname failed, ret=%d.", status);
    return false;
  }
  return true;
}

bool AsyncEventManager::RegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                   const EventProcessCallBack &cb) {
  if (cb == nullptr) {
    AICPU_LOGE("AsyncEventManager RegEventCb failed, cb is nullptr.");
    return false;
  }
  AsyncTaskInfo task_info;
  task_info.task_cb = cb;
  if (!GenTaskInfoFromCtx(&task_info)) {
    AICPU_LOGE("AsyncEventManager GenTaskInfoFromCtx failed.");
    return false;
  }
  AsyncEventInfo info;
  info.event_id = event_id;
  info.sub_event_id = sub_event_id;
  {
    std::unique_lock<std::mutex> _lock(map_mutex_);
    auto iter = asyncTaskMap_.find(info);
    if (iter != asyncTaskMap_.end()) {
      AICPU_LOGE("AsyncEventManager RegEventCb failed.");
      return false;
    }
    asyncTaskMap_[info] = task_info;
  }

  AICPU_LOGI(
    "AsyncEventManager RegEventCb success, event_id[%u], subeventId[%u], taskId[%lu],"
    " streamId[%u], waitType[%u], waitId[%u], opName[%s], startTick[%lu].",
    event_id, sub_event_id, task_info.task_id, task_info.stream_id, task_info.wait_type, task_info.wait_id,
    task_info.op_name.c_str(), task_info.start_tick);
  return true;
}

void AsyncEventManager::ProcessEvent(const uint32_t event_id, const uint32_t sub_event_id, void *param) {
  AICPU_LOGI("AsyncEventManager proc event_id = %d, sub_event_id = %d", event_id, sub_event_id);
  AsyncEventInfo info;
  info.event_id = event_id;
  info.sub_event_id = sub_event_id;
  EventProcessCallBack taskCb = nullptr;
  {
    std::unique_lock<std::mutex> lk(map_mutex_);
    auto iter = asyncTaskMap_.find(info);
    if (iter == asyncTaskMap_.end()) {
      AICPU_LOGW("AsyncEventManager no async task to deal with.");
      return;
    }
    taskCb = iter->second.task_cb;
    (void)asyncTaskMap_.erase(iter);
  }
  if (taskCb != nullptr) {
    taskCb(param);
  }
  AICPU_LOGI("AsyncEventManager proc end!");
  return;
}
}  // namespace aicpu

void AicpuNotifyWait(void *notify_param, const uint32_t param_len) {
  aicpu::AsyncEventManager::GetInstance().NotifyWait(notify_param, param_len);
  return;
}

bool AicpuRegEventCb(const uint32_t event_id, const uint32_t sub_event_id, const aicpu::EventProcessCallBack &cb) {
  return aicpu::AsyncEventManager::GetInstance().RegEventCb(event_id, sub_event_id, cb);
}
