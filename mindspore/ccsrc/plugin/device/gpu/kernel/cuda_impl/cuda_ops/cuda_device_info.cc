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
#ifndef _MSC_VER
#include <pthread.h>
#endif
#include <unordered_map>

namespace mindspore {
namespace device {
namespace gpu {
#ifndef _MSC_VER
pthread_rwlock_t GPUdeviceInfo::rwlock_;
#else
std::mutex GPUdeviceInfo::instanceLock;
#endif

#ifndef _MSC_VER
std::shared_ptr<GPUdeviceInfo> GPUdeviceInfo::GetInstance(uint32_t device_id) {
  static std::unordered_map<uint32_t, std::shared_ptr<GPUdeviceInfo>> instances;
  // read lock
  std::shared_ptr<GPUdeviceInfo> gpu_device_info{nullptr};
  pthread_rwlock_rdlock(&rwlock_);
  auto iter = instances.find(device_id);
  if (iter != instances.end()) {
    gpu_device_info = iter->second;
  }
  pthread_rwlock_unlock(&rwlock_);

  if (gpu_device_info == nullptr) {
    // write lock
    gpu_device_info = std::make_shared<GPUdeviceInfo>(device_id);
    pthread_rwlock_wrlock(&rwlock_);
    instances.emplace(device_id, gpu_device_info);
    pthread_rwlock_unlock(&rwlock_);
  }
  return gpu_device_info;
}
#else
std::shared_ptr<GPUdeviceInfo> GPUdeviceInfo::GetInstance(uint32_t device_id) {
  static std::unordered_map<uint32_t, std::shared_ptr<GPUdeviceInfo>> instances;
  std::shared_ptr<GPUdeviceInfo> gpu_device_info{nullptr};
  std::lock_guard<std::mutex> lk(instanceLock);
  auto iter = instances.find(device_id);
  if (iter != instances.end()) {
    gpu_device_info = iter->second;
  }
  if (gpu_device_info == nullptr) {
    gpu_device_info = std::make_shared<GPUdeviceInfo>(device_id);
    instances.emplace(device_id, gpu_device_info);
  }
  return gpu_device_info;
}
#endif

GPUdeviceInfo::GPUdeviceInfo(const uint32_t device_id) {
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  threads_per_block_ = prop.maxThreadsPerBlock;
  max_blocks_ = prop.multiProcessorCount;
  major_sm_ = prop.major;
  minor_sm_ = prop.minor;
  max_share_memory_ = prop.sharedMemPerBlock;
  const int x_index = 0;
  const int y_index = 1;
  const int z_index = 2;
  max_grid_size_.x = prop.maxGridSize[x_index];
  max_grid_size_.y = prop.maxGridSize[y_index];
  max_grid_size_.z = prop.maxGridSize[z_index];
#ifndef _MSC_VER
  pthread_rwlock_init(&rwlock_, nullptr);
#endif
}
GPUdeviceInfo::~GPUdeviceInfo() {
#ifndef _MSC_VER
  pthread_rwlock_destroy(&rwlock_);
#endif
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
