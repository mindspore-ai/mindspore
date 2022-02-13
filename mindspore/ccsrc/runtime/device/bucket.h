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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_BUCKET_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_BUCKET_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/device_event.h"
#include "runtime/device/launch_kernel.h"
#include "runtime/device/device_address.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore::device {
class Bucket {
 public:
  Bucket(uint32_t id, uint32_t bucket_size, std::string group, std::string device_name)
      : id_(id),
        bucket_size_(bucket_size),
        full_(false),
        stream_(nullptr),
        compute_stream_(nullptr),
        total_size_(0),
        device_id_(0),
        device_name_(std::move(device_name)),
        group_(std::move(group)),
        pre_event_(nullptr),
        post_event_(nullptr),
        launch_atomic_clean_(nullptr) {}
  virtual ~Bucket() = default;

  uint32_t id() const { return id_; }
  bool full() const { return full_; }
  void Launch();
  void Release();
  void AddGradTensor(const tensor::TensorPtr &tensor);
  virtual void Init(const std::vector<void *> &compute_streams, const std::vector<void *> &communication_streams) = 0;

 protected:
  uint32_t id_;
  uint32_t bucket_size_;
  bool full_;
  void *stream_;
  void *compute_stream_;
  size_t total_size_;
  uint32_t device_id_;
  std::string device_name_;
  std::string group_;

  std::shared_ptr<DeviceEvent> pre_event_;
  std::shared_ptr<DeviceEvent> post_event_;
  std::shared_ptr<LaunchKernel> launch_atomic_clean_;

  std::vector<DeviceAddressPtr> ar_input_address_list_;
  std::vector<DeviceAddressPtr> ar_output_address_list_;

  std::vector<size_t> align_size_list_;
  std::vector<tensor::TensorPtr> grad_tensor_list_;
  std::vector<uint8_t *> new_tensor_output_addrs_;
  std::vector<kernel::AddressPtr> memcpy_input_addrs_;
  std::vector<kernel::AddressPtr> memcpy_output_addrs_;
  std::vector<TypeId> tensor_type_list_;

  void UpdateTensorAddr();
  void AllocateAllReduceMemory();
  virtual void FreeAllDeviceMem() {}
  virtual void LaunchAllReduce() = 0;
  virtual void CopyTensorToContiguousMemory() = 0;
  virtual DeviceAddressPtr CreateDeviceAddress(size_t size, TypeId type_id, const std::string &format) const = 0;
  virtual size_t GetAlignSize(size_t size) const = 0;
  virtual void AllocateContinousMemory(const std::vector<DeviceAddressPtr> &to_allocate_address, size_t total_size,
                                       const std::vector<size_t> &size_list) const = 0;
};
}  // namespace mindspore::device

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_BUCKET_H_
