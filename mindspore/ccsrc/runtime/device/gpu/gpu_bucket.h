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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_BUCKET_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_BUCKET_H_

#include <memory>
#include <vector>
#include <string>
#include "runtime/device/bucket.h"

namespace mindspore::device::gpu {
class GPUBucket : public Bucket {
 public:
  GPUBucket(uint32_t id, uint32_t bucket_size);
  ~GPUBucket() override = default;

  void Init(const std::vector<void *> &compute_streams, const std::vector<void *> &communication_streams) override;

 protected:
  void CopyTensorToContiguousMemory() override;
  void LaunchAllReduce() override;
  DeviceAddressPtr CreateDeviceAddress(size_t size, TypeId type_id, const std::string &format) const override;
  size_t GetAlignSize(size_t size) const override;
  void AllocateContinousMemory(const std::vector<DeviceAddressPtr> &to_allocate_address, size_t total_size,
                               const std::vector<size_t> &size_list) const override;

  const void *collective_handle_;
};
}  // namespace mindspore::device::gpu
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_BUCKET_H_
