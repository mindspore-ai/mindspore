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
#include "runtime/device/gpu/gpu_device_address.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "runtime/device/gpu/gpu_device_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/gpu/distribution/collective_init.h"
#include "runtime/device/gpu/gpu_launch_mul.h"
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"
#include "runtime/device/gpu/gpu_common.h"
#include "runtime/hardware/device_context_manager.h"

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
GPUBucket::GPUBucket(uint32_t id, uint32_t bucket_size)
    : Bucket(id, bucket_size, kNcclWorldGroup, "GPU"), collective_handle_(nullptr) {}

DeviceAddressPtr GPUBucket::CreateDeviceAddress(size_t size, TypeId type_id, const std::string &format) const {
  return std::make_shared<GPUDeviceAddress>(nullptr, size, format, type_id, device_name_, device_id_);
}

size_t GPUBucket::GetAlignSize(size_t size) const { return AlignMemorySize(size); }

void GPUBucket::AllocateContinousMemory(const std::vector<DeviceAddressPtr> &to_allocate_address, size_t total_size,
                                        const std::vector<size_t> &size_list) const {
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  if (!device_context->AllocateContinuousMemory(to_allocate_address, total_size, size_list)) {
    MS_LOG(EXCEPTION) << "Allocate memory for AllReduce input failed";
  }
}

void GPUBucket::CopyTensorToContiguousMemory() {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(compute_stream_);
  if (ar_input_address_list_.empty()) {
    MS_LOG(EXCEPTION) << "AllReduce input address not found.";
  }
  MS_EXCEPTION_IF_NULL(ar_input_address_list_[0]);
  // Clean allreduce input
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(ar_input_address_list_[0]->GetMutablePtr(), 0, total_size_,
                                                     static_cast<cudaStream_t>(compute_stream_)),
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

  if (ar_input_address_list_.empty() || ar_output_address_list_.empty()) {
    MS_LOG(EXCEPTION) << "fusion AllReduce input address size is:" << ar_input_address_list_.size()
                      << " output address size is:" << ar_output_address_list_.size();
  }
  MS_EXCEPTION_IF_NULL(ar_input_address_list_[0]);
  MS_EXCEPTION_IF_NULL(ar_output_address_list_[0]);

  auto nccl_result = (*all_reduce_funcptr)(
    ar_input_address_list_[0]->GetMutablePtr(), ar_output_address_list_[0]->GetMutablePtr(), total_size_ / type_size,
    nccl_data_type_iter->second, ncclRedOp_t::ncclSum, static_cast<cudaStream_t>(stream_), group_);
  if (nccl_result != ncclSuccess) {
    MS_LOG(EXCEPTION) << "AllReduce failed, ret:" << nccl_result;
  }

  MS_LOG(INFO) << "end";
}

void GPUBucket::Init(const std::vector<void *> &compute_streams, const std::vector<void *> &communication_streams) {
  pre_event_ = std::make_shared<GpuEvent>();
  post_event_ = std::make_shared<GpuEvent>();

  if (!compute_streams.empty()) {
    compute_stream_ = compute_streams.front();
  }
  if (!communication_streams.empty()) {
    stream_ = communication_streams.front();
  }
  MS_EXCEPTION_IF_NULL(compute_stream_);
  MS_EXCEPTION_IF_NULL(stream_);

  MS_EXCEPTION_IF_NULL(pre_event_);
  MS_EXCEPTION_IF_NULL(post_event_);
  pre_event_->set_record_stream(compute_stream_);
  pre_event_->set_wait_stream(stream_);
  post_event_->set_record_stream(stream_);
  post_event_->set_wait_stream(compute_stream_);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
}
}  // namespace mindspore::device::gpu
