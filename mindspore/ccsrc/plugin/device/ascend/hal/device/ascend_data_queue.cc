/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include <string>
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace device {
void CheckRtRetWithError(rtError_t error, const std::string &msg) {
  if (error != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Rt error: " << msg << " | Error number: " << error;
  }
}

AscendDataQueueDynamic::AscendDataQueueDynamic(const size_t capacity)
    : DataQueue(capacity), stream_(nullptr), node_info_(nullptr) {
  auto context_key = device_context_->device_context_key();
  auto runtime_instance = dynamic_cast<ascend::AscendKernelRuntime *>(
    device::KernelRuntimeManager::Instance().GetKernelRuntime(context_key.device_name_, context_key.device_id_));
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  stream_ = runtime_instance->compute_stream();
}

BlockQueueStatus_T AscendDataQueueDynamic::Push(std::vector<DataQueueItem> data) {
  for (size_t i = 0; i < data.size(); i++) {
    auto &item = data[i];
    if (item.data_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Invalid Input: ptr: " << item.data_ptr_ << ", len: " << item.data_len_;
      return ERROR_INPUT;
    }
    void *addr = device_context_->device_res_manager_->AllocateMemory(item.data_len_);
    if (addr == nullptr) {
      MS_LOG(ERROR) << "Allocate device memory of data queue failed";
    }
    CheckRtRetWithError(
      rtMemcpyAsync(addr, item.data_len_, item.data_ptr_, item.data_len_, RT_MEMCPY_HOST_TO_DEVICE, stream_),
      "Rt Memcpy Error");
    item.device_addr_ = addr;
  }
  CheckRtRetWithError(rtStreamSynchronize(stream_), "Call runtime rtStreamSynchronize failed");
  node_info_[tail_].data_ = data;
  tail_ = (tail_ + 1) % (capacity_);
  ++size_;
  return SUCCESS;
}

BlockQueueStatus_T AscendDataQueueDynamic::Front(std::vector<DataQueueItem> *data) const {
  for (auto &item : node_info_[head_].data_) {
    host_release_(item.data_ptr_, item.worker_id_);
  }
  *data = node_info_[head_].data_;
  return SUCCESS;
}

BlockQueueStatus_T AscendDataQueueDynamic::Pop() {
  head_ = (head_ + 1) % (capacity_);
  --size_;
  return SUCCESS;
}

bool AscendDataQueueDynamic::Destroy() { return true; }
}  // namespace device
}  // namespace mindspore
