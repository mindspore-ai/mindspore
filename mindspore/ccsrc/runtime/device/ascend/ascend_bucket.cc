/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/ascend_bucket.h"

#include <vector>
#include <memory>
#include "runtime/mem.h"
#include "external/hccl/hccl.h"
#include "runtime/device/ascend/ascend_memory_pool.h"
#include "backend/kernel_compiler/hccl/hcom_util.h"
#include "backend/kernel_compiler/hccl/hccl_context.h"
#include "runtime/device/memory_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/ascend/ascend_event.h"
#include "runtime/device/ascend/ascend_launch_mul.h"
#include "runtime/device/ascend/ascend_launch_atomic_clean.h"
#include "utils/profile.h"

#define CHECK_ASCEND_RT_WITH_EXCEPTION(expression, message)    \
  {                                                            \
    rtError_t ret = (expression);                              \
    if (ret != RT_ERROR_NONE) {                                \
      MS_LOG(EXCEPTION) << message << ", error code: " << ret; \
    }                                                          \
  }

namespace mindspore::device::ascend {
void AscendBucket::AllocateAllReduceAddr() {
  // Check bucket is full
  if (grad_tensor_list_.size() != bucket_size_) {
    MS_LOG(EXCEPTION) << "grad tensor list size:" << grad_tensor_list_.size()
                      << " is not equal to bucket size:" << bucket_size_;
  }

  auto total_size = 0;
  std::vector<size_t> origin_size_list;
  for (auto &tensor : grad_tensor_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    tensor_type_list_.emplace_back(tensor->data_type());
    DeviceAddressPtr device_address = std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
    auto origin_size = device_address->GetSize();
    auto align_size = MemoryManager::GetCommonAlignSize(origin_size);
    origin_size_list.emplace_back(origin_size);
    align_size_list_.emplace_back(align_size);
    total_size += align_size;
    memcpy_input_addrs_.emplace_back(std::make_shared<kernel::Address>(
      static_cast<uint8_t *>(device_address->GetMutablePtr()), device_address->GetSize()));
  }

  total_size_ = total_size;

  auto runtime_instance = device::KernelRuntimeManager::Instance().GetCurrentKernelRuntime();
  MS_EXCEPTION_IF_NULL(runtime_instance);
  // AllReduce input output addr need to clear zero
  ar_input_addr_ = runtime_instance->MallocCommunicationMemFromMemPool(total_size);
  ar_output_addr_ = runtime_instance->MallocCommunicationMemFromMemPool(total_size);

  // generate memecpy output addr
  uint8_t *memcpy_output = ar_input_addr_;
  for (size_t i = 0; i < bucket_size_; ++i) {
    memcpy_output_addrs_.emplace_back(std::make_shared<kernel::Address>(memcpy_output, origin_size_list[i]));
    memcpy_output += align_size_list_[i];
  }
}

void AscendBucket::FreeDeviceMem(void *dev_ptr) { AscendMemoryPool::GetInstance().FreeTensorMem(dev_ptr); }

void AscendBucket::FreeAllDeviceMem() {
  if (ar_input_addr_ != nullptr) {
    uint8_t *origin_dev_addr = ar_input_addr_ - kMemAlignSize;
    FreeDeviceMem(origin_dev_addr);
    ar_input_addr_ = nullptr;
  }
  if (ar_output_addr_ != nullptr) {
    uint8_t *origin_dev_addr = ar_output_addr_ - kMemAlignSize;
    FreeDeviceMem(origin_dev_addr);
    ar_output_addr_ = nullptr;
  }
  // clear launch mul device Memory
  if (launch_mul_ != nullptr) {
    launch_mul_->FreeLaunchDeviceMem();
  }
  // clear launch atomic clean device Memory
  if (launch_atomic_clean_ != nullptr) {
    launch_atomic_clean_->FreeLaunchDeviceMem();
  }
}

void AscendBucket::CopyTensorToContiguousMemory() {
  // clear allreduce input addr
  CleanAllReduceInputAddr();
  for (size_t i = 0; i < bucket_size_; ++i) {
    MS_EXCEPTION_IF_NULL(memcpy_input_addrs_[i]);
    MS_EXCEPTION_IF_NULL(memcpy_output_addrs_[i]);
    MS_LOG(DEBUG) << "MemcpyAsync dst size:" << memcpy_output_addrs_[i]->size
                  << " src size:" << memcpy_input_addrs_[i]->size;
    if (memcpy_output_addrs_[i]->size < memcpy_input_addrs_[i]->size) {
      MS_LOG(EXCEPTION) << "rtMemcpyAsync dst size < src size";
    }

    CHECK_ASCEND_RT_WITH_EXCEPTION(
      rtMemcpyAsync(memcpy_output_addrs_[i]->addr, memcpy_output_addrs_[i]->size, memcpy_input_addrs_[i]->addr,
                    memcpy_input_addrs_[i]->size, RT_MEMCPY_DEVICE_TO_DEVICE, compute_stream_),
      "Call rtMemcpyAsync failed");
  }
}

void AscendBucket::LaunchAllReduce() {
  if (tensor_type_list_.empty()) {
    MS_LOG(EXCEPTION) << "No tesnor type found";
  }

  // AllReduce inputs data type should be same
  auto type = tensor_type_list_[0];
  if (std::any_of(tensor_type_list_.begin(), tensor_type_list_.end(),
                  [&type](TypeId tensor_type) { return type != tensor_type; })) {
    MS_LOG(EXCEPTION) << "allreduce input have different dtype";
  }

  auto iter = CONST_OP_HCOM_DATA_TYPE_MAP.find(type);
  if (iter == CONST_OP_HCOM_DATA_TYPE_MAP.end()) {
    MS_LOG(EXCEPTION) << "unknown data type:" << type;
  }

  uint32_t type_size;
  if (!HcomUtil::GetHcomTypeSize(iter->second, &type_size)) {
    MS_LOG(EXCEPTION) << "get hcom type size fialed";
  }

  if (type_size == 0 || total_size_ % type_size != 0) {
    MS_LOG(EXCEPTION) << "Total_size[" << total_size_ << "],Type_size[" << type_size << "] != 0, fail!";
  }
  auto hccl_count = total_size_ / type_size;

  HcclReduceOp op_type = HcclReduceOp::HCCL_REDUCE_SUM;
  auto hccl_result = HcclAllReduce(ar_input_addr_, ar_output_addr_, hccl_count, iter->second, op_type,
                                   kernel::HcclContext::GetInstance().hccl_comm(), stream_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "HcclAllReduce faled, ret:" << hccl_result;
  }
}

void AscendBucket::CleanAllReduceInputAddr() {
  if (launch_atomic_clean_ == nullptr) {
    launch_atomic_clean_ = CreateLaunchAtomicClean();
    MS_EXCEPTION_IF_NULL(launch_atomic_clean_);
  }
  // set atomic clean input addr
  launch_atomic_clean_->SetInputAddr(ar_input_addr_);
  // launch atomic clean
  launch_atomic_clean_->LaunchOpKernel();
}

std::shared_ptr<LaunchKernel> AscendBucket::CreateLaunchMul() {
  if (tensor_type_list_.empty()) {
    MS_LOG(ERROR) << "tensor_type_list_ is empty";
  }
  auto launch_mul = std::make_shared<AscendLaunchMul>(stream_, tensor_type_list_[0], total_size_);
  MS_EXCEPTION_IF_NULL(launch_mul);
  return launch_mul;
}

std::shared_ptr<LaunchKernel> AscendBucket::CreateLaunchAtomicClean() {
  if (tensor_type_list_.empty()) {
    MS_LOG(ERROR) << "tensor_type_list_ is empty";
  }
  auto launch_atomic_clean =
    std::make_shared<AscendLaunchAtomicClean>(compute_stream_, tensor_type_list_[0], total_size_);
  MS_EXCEPTION_IF_NULL(launch_atomic_clean);
  return launch_atomic_clean;
}

void AscendBucket::Init() {
  pre_event_ = std::make_shared<AscendEvent>();
  post_event_ = std::make_shared<AscendEvent>();

  auto kernel_runtime = KernelRuntimeManager::Instance().GetCurrentKernelRuntime();
  MS_EXCEPTION_IF_NULL(kernel_runtime);
  compute_stream_ = kernel_runtime->compute_stream();
  stream_ = kernel_runtime->communication_stream();

  MS_EXCEPTION_IF_NULL(pre_event_);
  MS_EXCEPTION_IF_NULL(post_event_);
  pre_event_->set_wait_stream(stream_);
  pre_event_->set_record_stream(compute_stream_);
  post_event_->set_wait_stream(compute_stream_);
  post_event_->set_record_stream(stream_);
}
}  // namespace mindspore::device::ascend
