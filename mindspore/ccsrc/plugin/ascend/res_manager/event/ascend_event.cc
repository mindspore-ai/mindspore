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

#include "plugin/ascend/res_manager/event/ascend_event.h"
#include <cstdint>
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"

namespace mindspore::device::ascend {
namespace {
std::string voidPtrToString(void *ptr) {
  uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  return std::to_string(address);
}
}  // namespace

class AscendTimeEvent : public AscendEvent {
 public:
  AscendTimeEvent();
  ~AscendTimeEvent() override = default;
};

AscendEvent::AscendEvent(uint32_t flag, bool use_extensional_api) {
  aclError ret;
  if (use_extensional_api) {
    ret = CALL_ASCEND_API(aclrtCreateEventExWithFlag, &event_, flag);
  } else {
    ret = CALL_ASCEND_API(aclrtCreateEventWithFlag, &event_, flag);
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtCreateEventExWithFlag failed, ret:" << ret;
    event_ = nullptr;
  }
  has_flag_ = true;
  MS_LOG(DEBUG) << "Create ascend event success, flag : " << flag << ".";
}

AscendTimeEvent::AscendTimeEvent() : AscendEvent(ACL_EVENT_TIME_LINE, false) {}

AscendEvent::~AscendEvent() {
  if (!event_destroyed_) {
    auto ret = CALL_ASCEND_API(aclrtDestroyEvent, event_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "aclrtDestroyEvent failed, ret:" << ret;
    }
  }

  event_ = nullptr;
  wait_stream_ = nullptr;
  record_stream_ = nullptr;
}

bool AscendEvent::IsReady() const { return event_ != nullptr; }

void AscendEvent::RecordEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(record_stream_);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Event", "RecordEvent", "");
    tracker::CALL_MEMORY_TRACKER(
      UpdateTask, "Event",
      {{tracker::kStreamId, std::to_string(AscendStreamMng::GetInstance().GetStreamId(record_stream_))},
       {tracker::kEvent, voidPtrToString(event_)}});
  }
  auto ret = CALL_ASCEND_API(aclrtRecordEvent, event_, record_stream_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtRecordEvent failed, ret:" << ret;
  }
  need_wait_ = true;
}

void AscendEvent::RecordEvent(uint32_t stream_id) {
  MS_LOG(DEBUG) << "Ascend record event on stream id : " << stream_id << ".";
  MS_EXCEPTION_IF_NULL(event_);
  record_stream_ = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(record_stream_);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Event", "RecordEvent", "");
    tracker::CALL_MEMORY_TRACKER(
      UpdateTask, "Event",
      {{tracker::kStreamId, std::to_string(stream_id)}, {tracker::kEvent, voidPtrToString(event_)}});
  }
  auto ret = CALL_ASCEND_API(aclrtRecordEvent, event_, record_stream_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtRecordEvent failed, ret:" << ret;
  }
  need_wait_ = true;
}

void AscendEvent::WaitEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(wait_stream_);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Event", "WaitEvent", "");
    tracker::CALL_MEMORY_TRACKER(
      UpdateTask, "Event",
      {{tracker::kStreamId, std::to_string(AscendStreamMng::GetInstance().GetStreamId(wait_stream_))},
       {tracker::kEvent, voidPtrToString(event_)}});
  }
  auto ret = CALL_ASCEND_API(aclrtStreamWaitEvent, wait_stream_, event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtStreamWaitEvent failed, ret:" << ret;
  }
  if (!has_flag_) {
    // The event created by aclrtCreateEventExWithFlag is not support to call
    // aclrtResetEvent/aclrtQueryEvent/aclrtQueryEventWaitStatus.
    MS_LOG(DEBUG) << "Reset Event";
    ret = CALL_ASCEND_API(aclrtResetEvent, event_, wait_stream_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "aclrtResetEvent failed, ret:" << ret;
    }
  }
  need_wait_ = false;
}

bool AscendEvent::WaitEvent(uint32_t stream_id) {
  MS_LOG(DEBUG) << "Ascend wait event on stream id : " << stream_id << ".";
  wait_stream_ = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Event", "WaitEvent", "");
    tracker::CALL_MEMORY_TRACKER(
      UpdateTask, "Event",
      {{tracker::kStreamId, std::to_string(stream_id)}, {tracker::kEvent, voidPtrToString(event_)}});
  }
  auto ret = CALL_ASCEND_API(aclrtStreamWaitEvent, wait_stream_, event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtStreamWaitEvent failed, ret:" << ret;
  }
  if (!has_flag_) {
    // Reset event after wait so that event can be reused.
    ret = CALL_ASCEND_API(aclrtResetEvent, event_, wait_stream_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "aclrtResetEvent failed, ret:" << ret;
    }
  }
  need_wait_ = false;
  return true;
}

void AscendEvent::WaitEventWithoutReset() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(wait_stream_);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Event", "WaitEvent", "");
    tracker::CALL_MEMORY_TRACKER(
      UpdateTask, "Event",
      {{tracker::kStreamId, std::to_string(AscendStreamMng::GetInstance().GetStreamId(wait_stream_))},
       {tracker::kEvent, voidPtrToString(event_)}});
  }
  // Query result will be reset after aclrtResetEvent is called.
  auto ret = CALL_ASCEND_API(aclrtStreamWaitEvent, wait_stream_, event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtStreamWaitEvent failed, ret:" << ret;
  }
  need_wait_ = false;
}

void AscendEvent::WaitEventWithoutReset(uint32_t stream_id) {
  wait_stream_ = AscendStreamMng::GetInstance().GetStream(stream_id);
  WaitEventWithoutReset();
}

void AscendEvent::ResetEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(wait_stream_);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Event", "ResetEvent", "");
    tracker::CALL_MEMORY_TRACKER(
      UpdateTask, "Event",
      {{tracker::kStreamId, std::to_string(AscendStreamMng::GetInstance().GetStreamId(wait_stream_))},
       {tracker::kEvent, voidPtrToString(event_)}});
  }

  MS_LOG(DEBUG) << "Reset Event";
  auto ret = CALL_ASCEND_API(aclrtResetEvent, event_, wait_stream_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtResetEvent failed, ret:" << ret;
  }
}

void AscendEvent::ResetEvent(uint32_t stream_id) {
  wait_stream_ = AscendStreamMng::GetInstance().GetStream(stream_id);
  ResetEvent();
}

void AscendEvent::SyncEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  auto ret = CALL_ASCEND_API(aclrtSynchronizeEvent, event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtSynchronizeEvent failed, ret:" << ret;
  }
}

bool AscendEvent::QueryEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  if (event_destroyed_) {
    MS_LOG(EXCEPTION) << "The event " << this
                      << " has been destroyed, cannot query it, please check the wait() function is called when you "
                         "use the async communication ops.";
  }
  aclrtEventRecordedStatus status;
  auto ret = CALL_ASCEND_API(aclrtQueryEventStatus, event_, &status);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclQueryEventStatus failed, ret:" << ret;
  }
  return status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
}

void AscendEvent::ElapsedTime(float *cost_time, const DeviceEvent *other) {
  MS_EXCEPTION_IF_NULL(event_);
  auto ascend_other = static_cast<const AscendEvent *>(other);
  MS_EXCEPTION_IF_NULL(ascend_other);
  MS_EXCEPTION_IF_NULL(ascend_other->event_);
  auto ret = CALL_ASCEND_API(aclrtEventElapsedTime, cost_time, event_, ascend_other->event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "aclrtEventElapsedTime failed, ret:" << ret;
  }
}

bool AscendEvent::NeedWait() { return need_wait_; }

bool AscendEvent::DestroyEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  auto ret = CALL_ASCEND_API(aclrtDestroyEvent, event_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtDestroyEvent failed, ret:" << ret;
  }
  event_destroyed_ = true;
  return true;
}
}  // namespace mindspore::device::ascend
