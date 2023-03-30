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

#ifndef AICPU_OPS_AICPU_ASYNC_EVENT_H_
#define AICPU_OPS_AICPU_ASYNC_EVENT_H_

#include <functional>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include "aicpu_sharder/aicpu_context.h"

namespace aicpu {
using NotifyFunc = std::function<void(void *param, const uint32_t param_len)>;
using EventProcessCallBack = std::function<void(void *param)>;

struct AsyncEventInfo {
  uint32_t event_id;
  uint32_t sub_event_id;

  bool operator==(const AsyncEventInfo &info) const {
    return (event_id == info.event_id) && (sub_event_id == info.sub_event_id);
  }
};

inline bool operator<(const AsyncEventInfo &info1, const AsyncEventInfo &info2) {
  return (info1.event_id < info2.event_id) ||
         ((info1.event_id == info2.event_id) && (info1.sub_event_id < info2.sub_event_id));
}

struct AsyncTaskInfo {
  uint64_t start_tick;
  std::string op_name;
  uint8_t wait_type;
  uint32_t wait_id;
  uint64_t task_id;
  uint32_t stream_id;
  EventProcessCallBack task_cb;
};

struct AsyncNotifyInfo {
  uint8_t wait_type;
  uint32_t wait_id;
  uint64_t task_id;
  uint32_t stream_id;
  uint32_t ret_code;
  aicpu::aicpuContext_t ctx;
};

class AsyncEventManager {
 public:
  /**
   * Get the unique object of this class
   */
  static AsyncEventManager &GetInstance();

  /**
   * Register notify callback function
   * @param notify wait notify callback function
   */
  void Register(const NotifyFunc &notify);

  /**
   * Notify wait task
   * @param notify_param notify param info
   * @param param_len notify_param len
   */
  void NotifyWait(void *notify_param, const uint32_t param_len);

  /**
   * Register Event callback function, async op call
   * @param eventID EventId
   * @param sub_event_id queue id
   * @param cb Event callback function
   * @return whether register success
   */
  bool RegEventCb(const uint32_t event_id, const uint32_t sub_event_id, const EventProcessCallBack &cb);

  /**
   * Process event
   * @param event_id EventId
   * @param sub_event_id queue id
   * @param param event param
   */
  void ProcessEvent(const uint32_t event_id, const uint32_t sub_event_id, void *param = nullptr);

 private:
  AsyncEventManager() : notify_func_(nullptr) {}
  ~AsyncEventManager() = default;

  AsyncEventManager(const AsyncEventManager &) = delete;
  AsyncEventManager &operator=(const AsyncEventManager &) = delete;
  AsyncEventManager(AsyncEventManager &&) = delete;
  AsyncEventManager &operator=(AsyncEventManager &&) = delete;

  // generate task info from ctx
  bool GenTaskInfoFromCtx(AsyncTaskInfo *task_info) const;

  // wait notify function
  NotifyFunc notify_func_;

  std::mutex map_mutex_;

  std::map<AsyncEventInfo, AsyncTaskInfo> asyncTaskMap_;
};
}  // namespace aicpu

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Notify wait task
 * @param notify_param notify info
 * @param param_len
 */
__attribute__((weak)) void AicpuNotifyWait(void *notify_param, const uint32_t param_len);

/**
 * Register Event callback function, async op call
 * @param info Registered event information
 * @param cb Event callback function
 * @return whether register success
 */
__attribute__((weak)) bool AicpuRegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                           const aicpu::EventProcessCallBack &cb);

#ifdef __cplusplus
}
#endif
#endif  // AICPU_OPS_AICPU_ASYNC_EVENT_H_
