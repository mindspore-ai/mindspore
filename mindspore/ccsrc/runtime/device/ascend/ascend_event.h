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

#ifndef MINDSPORE_ASCEND_EVENT_H
#define MINDSPORE_ASCEND_EVENT_H

#include "runtime/base.h"
#include "ir/device_event.h"
namespace mindspore::device::ascend {
class AscendEvent : public DeviceEvent {
 public:
  AscendEvent();
  ~AscendEvent() override;

  void WaitEvent() override;
  void RecordEvent() override;
  bool NeedWait() override;
  void set_wait_stream(rtStream_t wait_stream) override { wait_stream_ = wait_stream; }
  void set_record_stream(rtStream_t record_stream) override { record_stream_ = record_stream; }

 private:
  rtEvent_t event_{nullptr};
  rtStream_t wait_stream_{nullptr};
  rtStream_t record_stream_{nullptr};
  bool need_wait_{false};
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_ASCEND_EVENT_H
