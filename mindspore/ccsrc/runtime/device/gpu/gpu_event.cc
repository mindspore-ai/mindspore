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

#include "runtime/device/gpu/gpu_event.h"
#include "runtime/device/gpu/gpu_common.h"

namespace mindspore::device::gpu {
GpuEvent::GpuEvent() {
  auto ret = cudaEventCreate(&event_);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventCreate failed, ret:" << ret;
    event_ = nullptr;
  }
}

GpuEvent::~GpuEvent() { CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaEventDestroy(event_), "cudaEventDestory failed"); }

void GpuEvent::WaitEvent() {
  MS_EXCEPTION_IF_NULL(wait_stream_);
  MS_EXCEPTION_IF_NULL(event_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamWaitEvent(wait_stream_, event_, 0), "cudaStreamWaitEvent failed");
  need_wait_ = false;
}

void GpuEvent::RecordEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(record_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaEventRecord(event_, record_stream_), "cudaEventRecord failed");
  need_wait_ = true;
}

void GpuEvent::SyncEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaEventSynchronize(event_), "cudaEventSynchronize failed");
}

void GpuEvent::ElapsedTime(float *cost_time, const DeviceEvent *other) {
  MS_EXCEPTION_IF_NULL(event_);
  auto gpu_event = static_cast<const GpuEvent *>(other);
  MS_EXCEPTION_IF_NULL(gpu_event);
  MS_EXCEPTION_IF_NULL(gpu_event->event_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaEventElapsedTime(cost_time, event_, gpu_event->event_),
                                     "cudaEventElapsedTime failed");
}

bool GpuEvent::NeedWait() { return need_wait_; }
}  // namespace mindspore::device::gpu
