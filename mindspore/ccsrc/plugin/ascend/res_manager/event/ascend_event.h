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

#include "runtime/hardware_abstract/event/device_event.h"
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore::device::ascend {
constexpr uint32_t ACL_EVENT_DEFAULT = 0x0000000Eu;

class ASCEND_RES_MANAGER_EXPORT AscendEvent : public DeviceEvent {
 public:
  AscendEvent() = delete;
  explicit AscendEvent(uint32_t flag, bool use_extensional_api = true);
  ~AscendEvent() override;

  bool IsReady() const override;
  void WaitEvent() override;
  bool WaitEvent(uint32_t stream_id) override;
  void WaitEventWithoutReset() override;
  void WaitEventWithoutReset(uint32_t stream_id) override;

  void ResetEvent() override;
  void ResetEvent(uint32_t stream_id) override;

  void RecordEvent() override;
  void RecordEvent(uint32_t stream_id) override;
  bool NeedWait() override;
  void SyncEvent() override;
  bool QueryEvent() override;
  void ElapsedTime(float *cost_time, const DeviceEvent *other) override;
  bool DestroyEvent() override;
  void set_wait_stream(aclrtStream wait_stream) override { wait_stream_ = wait_stream; }
  void set_record_stream(aclrtStream record_stream) override { record_stream_ = record_stream; }

 protected:
  aclrtEvent event_{nullptr};
  aclrtStream wait_stream_{nullptr};
  aclrtStream record_stream_{nullptr};
  bool need_wait_{false};
  bool event_destroyed_{false};
  bool has_flag_{false};
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_ASCEND_EVENT_H
