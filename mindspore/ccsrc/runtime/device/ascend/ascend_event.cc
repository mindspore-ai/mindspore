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

#include "runtime/device/ascend/ascend_event.h"

#include "runtime/event.h"
#include "runtime/stream.h"
#include "utils/log_adapter.h"

namespace mindspore::device::ascend {
AscendEvent::AscendEvent() {
  auto ret = rtEventCreate(&event_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtEventCreate failed, ret:" << ret;
    event_ = nullptr;
  }
}

AscendTimeEvent::AscendTimeEvent() {
  auto ret = rtEventCreateWithFlag(&event_, RT_EVENT_TIME_LINE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtEventCreate failed, ret:" << ret;
    event_ = nullptr;
  }
}

AscendEvent::~AscendEvent() {
  auto ret = rtEventDestroy(event_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtEventDestory failed, ret:" << ret;
  }
}

void AscendEvent::RecordEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(record_stream_);
  auto ret = rtEventRecord(event_, record_stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtEventRecord failed, ret:" << ret;
  }
  need_wait_ = true;
}

void AscendEvent::WaitEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(wait_stream_);
  auto ret = rtStreamWaitEvent(wait_stream_, event_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtStreamWaitEvent failed, ret:" << ret;
  }
  ret = rtEventReset(event_, wait_stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtEventReset failed, ret:" << ret;
  }
  need_wait_ = false;
}

void AscendEvent::SyncEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  auto ret = rtEventSynchronize(event_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtEventSynchronize failed, ret:" << ret;
  }
}

void AscendEvent::ElapsedTime(float *cost_time, const DeviceEvent *other) {
  MS_EXCEPTION_IF_NULL(event_);
  auto ascend_other = static_cast<const AscendEvent *>(other);
  MS_EXCEPTION_IF_NULL(ascend_other);
  MS_EXCEPTION_IF_NULL(ascend_other->event_);
  auto ret = rtEventElapsedTime(cost_time, event_, ascend_other->event_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtEventElapsedTime failed, ret:" << ret;
  }
}

bool AscendEvent::NeedWait() { return need_wait_; }
}  // namespace mindspore::device::ascend
