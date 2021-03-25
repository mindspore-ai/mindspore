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

#include "runtime/device/gpu/gpu_bucket.h"

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <vector>
#include <memory>
#include "abstract/utils.h"
#include "runtime/device/gpu/gpu_event.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "runtime/device/gpu/gpu_device_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/gpu/distribution/collective_init.h"
#include "runtime/device/gpu/gpu_launch_mul.h"
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"
#include "runtime/device/gpu/gpu_common.h"

namespace {
const size_t kCommunicationMemAlignSize = 16;
size_t AlignMemorySize(size_t size) {
  if (size == 0) {
    return kCommunicationMemAlignSize;
  }
  return ((size + kCommunicationMemAlignSize - 1) / kCommunicationMemAlignSize) * kCommunicationMemAlignSize;
}
}  // namespace
namespace mindspore::device::gpu {
GPUBucket::GPUBucket(uint32_t id, uint32_t bucket_size) : Bucket(id, bucket_size), collective_handle_(nullptr) {
  group_ = kNcclWorldGroup;
}

void GPUBucket::AllocateAllReduceAddr() {
  MS_LOG(INFO) << "start";
  if (grad_tensor_list_.size() != bucket_size_) {
    MS_LOG(EXCEPTION) << "grad tensor list size:" << grad_tensor_list_.size()
                      << " is not equal to bucket size:" << bucket_size_;
  }

  auto total_size = 0;
  std::vector<size_t> size_list;
  for (auto &tensor : grad_tensor_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    tensor_type_list_.emplace_back(tensor->data_type());
    DeviceAddressPtr device_address = std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
    MS_EXCEPTION_IF_NULL(device_address);
    auto origin_size = device_address->GetSize();
    auto align_size = AlignMemorySize(origin_size);
    size_list.emplace_back(origin_size);
    align_size_list_.emplace_back(align_size);
    total_size += align_size;
    memcpy_input_addrs_.emplace_back(
      std::make_shared<kernel::Address>(static_cast<uint8_t *>(device_address->GetMutablePtr()), origin_size));
  }
  total_size_ = total_size;

  ar_input_addr_ = static_cast<uint8_t *>(GPUMemoryAllocator::GetInstance().AllocTensorMem(total_size));
  ar_output_addr_ = static_cast<uint8_t *>(GPUMemoryAllocator::GetInstance().AllocTensorMem(total_size));

  uint8_t *memcpy_output = ar_input_addr_;
  for (size_t i = 0; i < bucket_size_; ++i) {
    memcpy_output_addrs_.emplace_back(std::make_shared<kernel::Address>(memcpy_output, size_list[i]));
    memcpy_output += align_size_list_[i];
  }
  MS_LOG(INFO) << "end";
}

void GPUBucket::FreeDeviceMem(void *dev_ptr) { GPUMemoryAllocator::GetInstance().FreeTensorMem(dev_ptr); }

void GPUBucket::FreeAllDeviceMem() {
  MS_LOG(INFO) << "start";
  if (ar_input_addr_ != nullptr) {
    FreeDeviceMem(ar_input_addr_);
    ar_input_addr_ = nullptr;
  }
  if (ar_output_addr_ != nullptr) {
    FreeDeviceMem(ar_output_addr_);
    ar_output_addr_ = nullptr;
  }
  // clear launch mul device memory
  if (launch_mul_ != nullptr) {
    launch_mul_->FreeLaunchDeviceMem();
  }
  MS_LOG(INFO) << "end";
}

void GPUBucket::CopyTensorToContiguousMemory() {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(compute_stream_);
  // Clean allreduce input
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(ar_input_addr_, 0, total_size_, static_cast<cudaStream_t>(compute_stream_)),
    "Call cudaMemsetAsync failed");

  for (size_t i = 0; i < bucket_size_; ++i) {
    MS_EXCEPTION_IF_NULL(memcpy_output_addrs_[i]);
    MS_EXCEPTION_IF_NULL(memcpy_input_addrs_[i]);
    if (!GPUDeviceManager::GetInstance().CopyDeviceMemToDeviceAsync(memcpy_output_addrs_[i]->addr,
                                                                    memcpy_input_addrs_[i]->addr,
                                                                    memcpy_output_addrs_[i]->size, compute_stream_)) {
      MS_LOG(EXCEPTION) << "Copy memory failed";
    }
  }
  MS_LOG(INFO) << "end";
}

void GPUBucket::LaunchAllReduce() {
  MS_LOG(INFO) << "start";
  collective_handle_ = device::gpu::CollectiveInitializer::instance().collective_handle();
  auto all_reduce_funcptr =
    reinterpret_cast<kernel::AllReduce>(dlsym(const_cast<void *>(collective_handle_), "AllReduce"));
  MS_EXCEPTION_IF_NULL(all_reduce_funcptr);
  MS_EXCEPTION_IF_NULL(stream_);

  if (tensor_type_list_.empty()) {
    MS_LOG(EXCEPTION) << "No tesnor type found";
  }
  auto type = tensor_type_list_[0];
  if (std::any_of(tensor_type_list_.begin(), tensor_type_list_.end(),
                  [&type](TypeId tensor_type) { return type != tensor_type; })) {
    MS_LOG(EXCEPTION) << "AllReduce input have different dtype";
  }

  auto type_size = abstract::TypeIdSize(type);
  if (type_size == 0) {
    MS_LOG(EXCEPTION) << "Invalid type:" << type;
  }

  // typeid to nccl_data_type
  auto nccl_data_type_iter = kernel::kNcclDtypeMap.find(TypeIdLabel(type));
  if (nccl_data_type_iter == kernel::kNcclDtypeMap.end()) {
    MS_LOG(EXCEPTION) << "Invalid type:" << type;
  }

  auto nccl_result =
    (*all_reduce_funcptr)(ar_input_addr_, ar_output_addr_, total_size_ / type_size, nccl_data_type_iter->second,
                          ncclRedOp_t::ncclSum, static_cast<cudaStream_t>(stream_), group_);
  if (nccl_result != ncclSuccess) {
    MS_LOG(EXCEPTION) << "AllReduce failed, ret:" << nccl_result;
  }

  MS_LOG(INFO) << "end";
}

std::shared_ptr<LaunchKernel> GPUBucket::CreateLaunchMul() {
  if (tensor_type_list_.empty()) {
    MS_LOG(ERROR) << "tensor_type_list_ is empty";
  }
  auto launch_mul = std::make_shared<GPULaunchMul>(stream_, tensor_type_list_[0], total_size_);
  MS_EXCEPTION_IF_NULL(launch_mul);
  return launch_mul;
}

void GPUBucket::Init() {
  pre_event_ = std::make_shared<GpuEvent>();
  post_event_ = std::make_shared<GpuEvent>();

  auto kernel_runtime = KernelRuntimeManager::Instance().GetCurrentKernelRuntime();
  MS_EXCEPTION_IF_NULL(kernel_runtime);
  stream_ = kernel_runtime->communication_stream();
  compute_stream_ = kernel_runtime->compute_stream();

  MS_EXCEPTION_IF_NULL(pre_event_);
  MS_EXCEPTION_IF_NULL(post_event_);
  pre_event_->set_record_stream(compute_stream_);
  pre_event_->set_wait_stream(stream_);
  post_event_->set_record_stream(stream_);
  post_event_->set_wait_stream(compute_stream_);
}
}  // namespace mindspore::device::gpu
