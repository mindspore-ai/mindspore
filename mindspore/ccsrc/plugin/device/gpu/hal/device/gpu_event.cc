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

#include "plugin/device/gpu/hal/device/gpu_event.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"

namespace mindspore::device::gpu {
GpuEvent::GpuEvent() {
  auto ret = cudaEventCreate(&event_);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventCreate failed, ret:" << ret;
    event_ = nullptr;
  }
}

GpuEvent::GpuEvent(uint32_t flag) {
  auto ret = cudaEventCreateWithFlags(&event_, flag);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventCreateWithFlags failed, ret:" << ret;
    event_ = nullptr;
  }
}

GpuEvent::~GpuEvent() {
  if (!event_destroyed_) {
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaEventDestroy(event_), "cudaEventDestory failed");
  }

  event_ = nullptr;
  wait_stream_ = nullptr;
  record_stream_ = nullptr;
}

bool GpuEvent::IsReady() const { return event_ != nullptr; }

void GpuEvent::WaitEvent() {
  MS_EXCEPTION_IF_NULL(wait_stream_);
  MS_EXCEPTION_IF_NULL(event_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamWaitEvent(wait_stream_, event_, 0), "cudaStreamWaitEvent failed");
  need_wait_ = false;
}

bool GpuEvent::WaitEvent(uint32_t stream_id) {
  MS_LOG(DEBUG) << "Gpu wait event on stream id : " << stream_id << ".";
  MS_EXCEPTION_IF_NULL(event_);
  wait_stream_ = static_cast<cudaStream_t>(GPUDeviceManager::GetInstance().GetStream(stream_id));
  MS_EXCEPTION_IF_NULL(wait_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamWaitEvent(wait_stream_, event_, 0), "cudaStreamWaitEvent failed");
  need_wait_ = false;
  return true;
}

void GpuEvent::WaitEventWithoutReset() {
  // No reset api in cuda, so WaitEventWithoutReset is same with WaitEvent
  WaitEvent();
}

void GpuEvent::RecordEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(record_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaEventRecord(event_, record_stream_), "cudaEventRecord failed");
  need_wait_ = true;
}

void GpuEvent::RecordEvent(uint32_t stream_id) {
  MS_LOG(DEBUG) << "Gpu record event on stream id : " << stream_id << ".";
  MS_EXCEPTION_IF_NULL(event_);
  record_stream_ = static_cast<cudaStream_t>(GPUDeviceManager::GetInstance().GetStream(stream_id));
  MS_EXCEPTION_IF_NULL(record_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaEventRecord(event_, record_stream_), "cudaEventRecord failed");
  need_wait_ = true;
}

void GpuEvent::SyncEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaEventSynchronize(event_), "cudaEventSynchronize failed");
}

bool GpuEvent::QueryEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  return (cudaEventQuery(event_) == cudaSuccess) ? true : false;
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

bool GpuEvent::DestroyEvent() {
  MS_EXCEPTION_IF_NULL(event_);
  auto ret = cudaEventDestroy(event_);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaEventDestroy failed, ret: " << ret;
    return false;
  }
  event_destroyed_ = true;
  return true;
}
}  // namespace mindspore::device::gpu
