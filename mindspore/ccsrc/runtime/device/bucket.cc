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

#include "runtime/device/bucket.h"

#include "runtime/device/kernel_runtime_manager.h"
#include "include/common/utils/parallel_context.h"
#include "utils/profile.h"

namespace mindspore::device {
void Bucket::AddGradTensor(const tensor::TensorPtr &tensor) {
  if (grad_tensor_list_.size() >= bucket_size_) {
    MS_LOG(EXCEPTION) << "bucket is full";
  }
  grad_tensor_list_.emplace_back(tensor);
  if (grad_tensor_list_.size() > bucket_size_) {
    MS_LOG(EXCEPTION) << "too many tensor add to the bucket, bucket_size_:" << bucket_size_
                      << " total tensor size:" << grad_tensor_list_.size();
  }
  MS_LOG(INFO) << "current bucket tensors size:" << grad_tensor_list_.size();
  // bucket is full, start to launch allreduce
  if (grad_tensor_list_.size() == bucket_size_) {
    full_ = true;
  }
}

void Bucket::Launch() {
  auto start = GetTime();
  if (grad_tensor_list_.size() != bucket_size_) {
    MS_LOG(EXCEPTION) << "Bucket is not full, grad_tensor_list_ size:" << grad_tensor_list_.size()
                      << " bucket_size_:" << bucket_size_;
  }
  MS_LOG(INFO) << "Bucket is full, start to launch AllReduce";
  MS_EXCEPTION_IF_NULL(pre_event_);
  MS_EXCEPTION_IF_NULL(post_event_);
  AllocateAllReduceMemory();
  CopyTensorToContiguousMemory();
  pre_event_->RecordEvent();
  pre_event_->WaitEvent();
  LaunchAllReduce();
  post_event_->RecordEvent();
  post_event_->WaitEvent();
  UpdateTensorAddr();
  MS_LOG(INFO) << "Bucket launch cost:" << (GetTime() - start) * 1e6 << " us";
}

void Bucket::AllocateAllReduceMemory() {
  // Check bucket is full
  if (grad_tensor_list_.size() != bucket_size_) {
    MS_LOG(EXCEPTION) << "Grad tensor list size:" << grad_tensor_list_.size()
                      << " is not equal to bucket size:" << bucket_size_;
  }

  size_t total_size = 0;
  std::vector<size_t> origin_size_list;
  for (auto &tensor : grad_tensor_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    (void)tensor_type_list_.emplace_back(tensor->data_type());
    DeviceAddressPtr device_address = std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
    MS_EXCEPTION_IF_NULL(device_address);
    auto origin_size = device_address->GetSize();
    auto align_size = GetAlignSize(origin_size);
    (void)origin_size_list.emplace_back(origin_size);
    (void)align_size_list_.emplace_back(align_size);
    total_size += align_size;
    (void)memcpy_input_addrs_.emplace_back(std::make_shared<kernel::Address>(
      static_cast<uint8_t *>(device_address->GetMutablePtr()), device_address->GetSize()));

    auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    MS_EXCEPTION_IF_NULL(tensor_address);

    (void)ar_input_address_list_.emplace_back(
      CreateDeviceAddress(origin_size, tensor_address->type_id(), tensor_address->format()));
    (void)ar_output_address_list_.emplace_back(
      CreateDeviceAddress(origin_size, tensor_address->type_id(), tensor_address->format()));
  }

  total_size_ = total_size;

  AllocateContinuousMemory(ar_input_address_list_, total_size, align_size_list_);
  AllocateContinuousMemory(ar_output_address_list_, total_size, align_size_list_);

  // generate memecpy output addr
  if (origin_size_list.size() != ar_input_address_list_.size()) {
    MS_LOG(EXCEPTION) << "Invalid ar_input_address_list size:" << ar_input_address_list_.size()
                      << " origin_size_list size:" << origin_size_list.size();
  }
  size_t item_index = 0;
  for (const auto &ar_input_address_item : ar_input_address_list_) {
    MS_EXCEPTION_IF_NULL(ar_input_address_item);
    (void)memcpy_output_addrs_.emplace_back(
      std::make_shared<kernel::Address>(ar_input_address_item->GetMutablePtr(), origin_size_list[item_index]));
    ++item_index;
  }
}

void Bucket::UpdateTensorAddr() {
  if (grad_tensor_list_.size() != bucket_size_ || ar_output_address_list_.size() != bucket_size_) {
    MS_LOG(EXCEPTION) << "grad_tensor_list_ size:" << grad_tensor_list_.size()
                      << " ar_output_address_list_ size:" << ar_output_address_list_.size()
                      << " bucket size:" << bucket_size_;
  }

  for (size_t i = 0; i < bucket_size_; ++i) {
    auto &tensor = grad_tensor_list_[i];
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_device_address(ar_output_address_list_[i]);
  }
}

void Bucket::Release() {
  MS_LOG(INFO) << "Clear bucket:" << id_;
  grad_tensor_list_.clear();
  align_size_list_.clear();
  new_tensor_output_addrs_.clear();
  memcpy_input_addrs_.clear();
  memcpy_output_addrs_.clear();
  tensor_type_list_.clear();
  ar_input_address_list_.clear();
  ar_output_address_list_.clear();
  FreeAllDeviceMem();
  full_ = false;
}
}  // namespace mindspore::device
