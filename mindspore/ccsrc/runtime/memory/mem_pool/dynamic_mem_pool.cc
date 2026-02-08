/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "include/runtime/memory/mem_pool/dynamic_mem_pool.h"

#include <numeric>
#include <ostream>
#include <map>
#include <memory>
#include <string>
#include "utils/log_adapter.h"
#include "runtime/hardware_abstract/event/device_event.h"

namespace mindspore {
namespace device {
static thread_local AllocatorDebugInfo debug_info_;

AllocatorDebugInfo &DynamicMemAllocatorDebugInfo::GetDebugInfo() noexcept { return debug_info_; }

// Set the debug info when memory alloc.
void DynamicMemAllocatorDebugInfo::SetDebugInfo(const std::string &name, memory::mem_pool::MemType type,
                                                int input_index, int output_index, uint8_t run_mode) {
  debug_info_.name_ = name;
  debug_info_.type_ = type;
  debug_info_.input_index_ = input_index;
  debug_info_.output_index_ = output_index;
  debug_info_.run_mode_ = run_mode;
}

static const std::map<DynamicMemBufStatus, std::string> kBufStatusString = {
  {DynamicMemBufStatus::kMemBufIdle, "idle"},
  {DynamicMemBufStatus::kMemBufUsed, "used"},
  {DynamicMemBufStatus::kMemBufEagerFree, "eager_free"},
  {DynamicMemBufStatus::kMemBufUsedByEvent, "used_by_event"}};

const std::string &DynamicMemBufStatusToString(DynamicMemBufStatus status) { return kBufStatusString.at(status); }

bool EventBase::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id, const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (events_ == nullptr) {
    events_ = std::make_shared<std::unordered_map<uint32_t, std::shared_ptr<std::list<TaskIdOnStreamEvent>>>>();
  }
  std::shared_ptr<std::list<TaskIdOnStreamEvent>> event_list = nullptr;
  auto iter = events_->find(user_stream_id);
  if (iter == events_->end()) {
    event_list = std::make_shared<std::list<TaskIdOnStreamEvent>>();
    (void)events_->emplace(user_stream_id, event_list);
  } else {
    event_list = iter->second;
    MS_EXCEPTION_IF_NULL(event_list);
  }
  (void)event_list->emplace_back(task_id_on_stream, event);
  return true;
}

bool EventBase::WaitEvent(uint32_t task_id_on_stream, uint32_t user_stream_id) {
  if (events_ == nullptr) {
    return false;
  }
  auto iter = events_->find(user_stream_id);
  if (iter == events_->end()) {
    return false;
  }
  auto &event_list = iter->second;
  MS_EXCEPTION_IF_NULL(event_list);
  // Pop all element in list that not bigger than task_id_on_stream.
  while (!event_list->empty() && event_list->front().first <= task_id_on_stream) {
    event_list->pop_front();
  }
  // Remove list if event list is empty.
  if (event_list->empty()) {
    events_->erase(iter);
  }
  return true;
}

bool EventBase::IsEventNotUsed() { return events_ == nullptr ? true : events_->empty(); }

bool EventBase::SyncAllEvents() {
  if (IsEventNotUsed()) {
    return false;
  }

  for (auto iter = events_->begin(); iter != events_->end();) {
    auto &event_list = iter->second;
    MS_EXCEPTION_IF_NULL(event_list);
    for (auto list_iter = event_list->begin(); list_iter != event_list->end();) {
      auto &event = list_iter->second;
      // Sync event if event is not arrived.
      if (!event->QueryEvent()) {
        event->SyncEvent();
      }
      list_iter = event_list->erase(list_iter);
    }
    if (event_list->empty()) {
      // list is empty, erase list in map.
      iter = events_->erase(iter);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Event list is not empty.";
    }
  }
  return events_->empty();
}
}  // namespace device
}  // namespace mindspore
