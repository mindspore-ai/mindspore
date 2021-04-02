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
#include "backend/session/kernel_graph.h"

namespace mindspore::device {
class Bucket {
 public:
  Bucket(uint32_t id, uint32_t bucket_size)
      : id_(id),
        bucket_size_(bucket_size),
        full_(false),
        stream_(nullptr),
        compute_stream_(nullptr),
        pre_event_(nullptr),
        post_event_(nullptr),
        launch_mul_(nullptr),
        launch_atomic_clean_(nullptr),
        total_size_(0),
        ar_input_addr_(nullptr),
        ar_output_addr_(nullptr) {}
  virtual ~Bucket() = default;

  uint32_t id() const { return id_; }
  bool full() const { return full_; }
  void Launch();
  void Release();
  void AddGradTensor(const tensor::TensorPtr &tensor);
  virtual void Init() = 0;

 protected:
  uint32_t id_;
  uint32_t bucket_size_;
  bool full_;
  void *stream_;
  void *compute_stream_;

  std::shared_ptr<DeviceEvent> pre_event_;
  std::shared_ptr<DeviceEvent> post_event_;
  std::shared_ptr<LaunchKernel> launch_mul_;
  std::shared_ptr<LaunchKernel> launch_atomic_clean_;

  size_t total_size_;
  uint8_t *ar_input_addr_;
  uint8_t *ar_output_addr_;
  std::string group_;
  std::vector<size_t> align_size_list_;
  std::vector<tensor::TensorPtr> grad_tensor_list_;
  std::vector<uint8_t *> new_tensor_output_addrs_;
  std::vector<kernel::AddressPtr> memcpy_input_addrs_;
  std::vector<kernel::AddressPtr> memcpy_output_addrs_;
  std::vector<TypeId> tensor_type_list_;
  std::vector<void *> tensor_old_addr_list_;

  virtual void AllocateAllReduceAddr() = 0;
  void UpdateTensorAddr();
  void CalculateMean();
  virtual std::shared_ptr<LaunchKernel> CreateLaunchMul() = 0;
  virtual void LaunchAllReduce() = 0;
  virtual void FreeAllDeviceMem() = 0;
  virtual void FreeDeviceMem(void *dev_ptr) = 0;
  virtual void CopyTensorToContiguousMemory() = 0;
  void UpdateTensorOutputAddr(uint8_t *addr);
  void LazyDeleteOldAddr();
};
}  // namespace mindspore::device

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_BUCKET_H_
