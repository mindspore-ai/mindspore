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

#include "src/delegate/tensorrt/tensorrt_allocator.h"
#include <cuda_runtime.h>
#include <mutex>
#include "src/common/log_adapter.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
void *TensorRTAllocator::MallocDeviceMem(mindspore::MSTensor host_tensor, size_t size) {
  if (host_tensor == nullptr) {
    return nullptr;
  }
  if (cuda_tensor_map_.find(host_tensor.Name()) != cuda_tensor_map_.end()) {
    return nullptr;
  }

  auto cuda_type = ConvertDataType(host_tensor.DataType());
  if (static_cast<int>(cuda_type) == -1) {
    MS_LOG(ERROR) << "Unsupported Tensor Type:" << static_cast<int>(host_tensor.DataType());
    return nullptr;
  }
  void *device_ptr;
  auto cuda_ret = cudaMalloc(&device_ptr, size);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda Malloc failed for size:" << size;
    return nullptr;
  }
  cuda_tensor_map_[host_tensor.Name()] = device_ptr;
  return device_ptr;
}

void *TensorRTAllocator::GetDevicePtr(const std::string &tensor_name) {
  if (tensor_name.empty()) {
    return nullptr;
  }
  if (cuda_tensor_map_.find(tensor_name) == cuda_tensor_map_.end()) {
    return nullptr;
  }
  return this->cuda_tensor_map_.find(tensor_name)->second;
}

int TensorRTAllocator::SyncMemInHostAndDevice(mindspore::MSTensor host_tensor, const std::string &device_tensor_name,
                                              bool is_host2device, bool sync) {
  if (host_tensor == nullptr || host_tensor.Data() == nullptr ||
      cuda_tensor_map_.find(device_tensor_name) == cuda_tensor_map_.end()) {
    MS_LOG(ERROR) << " host or device ptr is null.";
    return RET_ERROR;
  }
  auto device_ptr = cuda_tensor_map_.find(device_tensor_name)->second;

  void *src_ptr = is_host2device ? host_tensor.MutableData() : device_ptr;
  void *dst_ptr = is_host2device ? device_ptr : host_tensor.MutableData();
  cudaMemcpyKind kind = is_host2device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  auto cuda_ret = cudaMemcpy(dst_ptr, src_ptr, host_tensor.DataSize(), kind);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "copy mem failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorRTAllocator::ClearDeviceMem() {
  for (const auto &iter : cuda_tensor_map_) {
    auto cuda_ret = cudaFree(iter.second);
    if (cuda_ret != cudaSuccess && cuda_ret != cudaErrorCudartUnloading) {
      MS_LOG(WARNING) << "free cuda failed for " << cudaGetErrorName(cuda_ret);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
