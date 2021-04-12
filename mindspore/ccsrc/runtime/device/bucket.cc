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

#include <memory>
#include "runtime/device/kernel_runtime_manager.h"
#include "frontend/parallel/context.h"
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
  AllocateAllReduceAddr();
  CopyTensorToContiguousMemory();
  pre_event_->RecordEvent();
  pre_event_->WaitEvent();
  LaunchAllReduce();
  // mul fusion
  CalculateMean();
  post_event_->RecordEvent();
  UpdateTensorAddr();
  // pass event to the tensor
  for (auto &tensor : grad_tensor_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->SetDeviceEvent(post_event_);
  }
  MS_LOG(INFO) << "Bucket launch cost:" << (GetTime() - start) * 1e6 << " us";
}

void Bucket::UpdateTensorAddr() {
  if (grad_tensor_list_.size() != bucket_size_ || new_tensor_output_addrs_.size() != bucket_size_) {
    MS_LOG(EXCEPTION) << "grad_tensor_list size:" << grad_tensor_list_.size()
                      << " tensor output addr size:" << new_tensor_output_addrs_.size()
                      << " bucket size:" << bucket_size_;
  }

  for (size_t i = 0; i < bucket_size_; ++i) {
    auto &tensor = grad_tensor_list_[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto device_address = std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
    // release old addr and manage addr by this Bucket.
    MS_EXCEPTION_IF_NULL(device_address);
    auto origin_dev_ptr = device_address->GetMutablePtr();
    tensor_old_addr_list_.emplace_back(origin_dev_ptr);
    device_address->from_mem_pool_ = false;
    device_address->set_ptr(new_tensor_output_addrs_[i]);
  }
}

void Bucket::CalculateMean() {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto grad_mean = parallel_context->gradients_mean();
  if (!grad_mean) {
    UpdateTensorOutputAddr(ar_output_addr_);
    return;
  }
  if (launch_mul_ == nullptr) {
    launch_mul_ = CreateLaunchMul();
    MS_EXCEPTION_IF_NULL(launch_mul_);
  }
  // set mul input1 addr
  launch_mul_->SetInputAddr(ar_output_addr_);
  // launch mean
  launch_mul_->LaunchOpKernel();
  // store tensor output addr
  auto launch_output = launch_mul_->GetKernelOutputAddr();
  if (launch_output.size() != 1) {
    MS_LOG(EXCEPTION) << "launch mul outputs should have one output";
  }
  UpdateTensorOutputAddr(launch_output[0]);
}

void Bucket::UpdateTensorOutputAddr(uint8_t *addr) {
  uint8_t *tensor_output = addr;
  for (size_t i = 0; i < bucket_size_; ++i) {
    new_tensor_output_addrs_.emplace_back(tensor_output);
    tensor_output += align_size_list_[i];
  }
}

void Bucket::LazyDeleteOldAddr() {
  MS_LOG(INFO) << "Lazy delete old grad address";
  for (auto old_addr : tensor_old_addr_list_) {
    FreeDeviceMem(old_addr);
  }
  tensor_old_addr_list_.clear();
}

void Bucket::Release() {
  MS_LOG(INFO) << "Clear bucket:" << id_;
  grad_tensor_list_.clear();
  align_size_list_.clear();
  new_tensor_output_addrs_.clear();
  memcpy_input_addrs_.clear();
  memcpy_output_addrs_.clear();
  tensor_type_list_.clear();
  LazyDeleteOldAddr();
  FreeAllDeviceMem();
  full_ = false;
}
}  // namespace mindspore::device
