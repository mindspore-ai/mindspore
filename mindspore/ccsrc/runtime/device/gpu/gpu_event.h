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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_EVENT_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_EVENT_H_

#include <cuda_runtime_api.h>
#include "ir/device_event.h"

namespace mindspore::device::gpu {
class GpuEvent : public DeviceEvent {
 public:
  GpuEvent();
  ~GpuEvent() override;

  void WaitEvent() override;
  void RecordEvent() override;
  bool NeedWait() override;
  void set_wait_stream(void *wait_stream) override { wait_stream_ = static_cast<cudaStream_t>(wait_stream); }
  void set_record_stream(void *record_stream) override { record_stream_ = static_cast<cudaStream_t>(record_stream); }

 private:
  cudaEvent_t event_{nullptr};
  cudaStream_t wait_stream_{nullptr};
  cudaStream_t record_stream_{nullptr};
  bool need_wait_{false};
};
}  // namespace mindspore::device::gpu
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_EVENT_H_
