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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include <unordered_map>
#include <mutex>

namespace mindspore {
namespace device {
namespace gpu {
std::shared_ptr<GPUdeviceInfo> GPUdeviceInfo::GetInstance(uint32_t device_id) {
  static std::unordered_map<uint32_t, std::shared_ptr<GPUdeviceInfo>> instances;
  static std::mutex mutex;
  std::lock_guard<std::mutex> _l(mutex);
  auto iter = instances.find(device_id);
  if (iter != instances.end()) {
    return iter->second;
  } else {
    auto gpu_device_info = std::make_shared<GPUdeviceInfo>(device_id);
    instances.emplace(device_id, gpu_device_info);
    return gpu_device_info;
  }
}

GPUdeviceInfo::GPUdeviceInfo(const uint32_t device_id) {
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  threads_per_block_ = prop.maxThreadsPerBlock;
  max_blocks_ = prop.multiProcessorCount;
  major_sm_ = prop.major;
  minor_sm_ = prop.minor;
  max_share_memory_ = prop.sharedMemPerBlock;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
