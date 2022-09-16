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

#include "plugin/device/gpu/kernel/rl/priority_replay_buffer_gpu_kernel.h"
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/priority_replay_buffer.h"

namespace mindspore {
namespace kernel {
using PriorityReplayBufferFactory = ReplayBufferFactory<PriorityReplayBuffer<SumMinTree>>;

PriorityReplayBufferCreateGpuKernel::~PriorityReplayBufferCreateGpuKernel() {
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  if (handle_device_) {
    allocator.FreeTensorMem(handle_device_);
  }
}

bool PriorityReplayBufferCreateGpuKernel::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::PriorityReplayBufferCreate>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast PriorityReplayBufferCreate ops failed!";
    return false;
  }

  const int64_t &capacity = kernel_ptr->get_capacity();
  const float &alpha = kernel_ptr->get_alpha();
  const std::vector<int64_t> &schema = kernel_ptr->get_schema();
  const int64_t &seed0 = kernel_ptr->get_seed0();
  const int64_t &seed1 = kernel_ptr->get_seed1();

  unsigned int seed = 0;
  std::random_device rd;
  if (seed1 != 0) {
    seed = static_cast<unsigned int>(seed1);
  } else if (seed0 != 0) {
    seed = static_cast<unsigned int>(seed0);
  } else {
    seed = rd();
  }

  std::vector<size_t> schema_in_size;
  std::transform(schema.begin(), schema.end(), std::back_inserter(schema_in_size),
                 [](const int64_t &arg) -> size_t { return LongToSize(arg); });

  auto &factory = PriorityReplayBufferFactory::GetInstance();
  std::tie(handle_, prioriory_replay_buffer_) = factory.Create(seed, alpha, capacity, schema_in_size);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);

  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  handle_device_ = static_cast<int64_t *>(allocator.AllocTensorMem(sizeof(handle_)));
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpy(handle_device_, &handle_, sizeof(handle_), cudaMemcpyHostToDevice),
                                    "cudaMemcpy failed.");

  output_size_list_.push_back(sizeof(handle_));
  return true;
}

std::vector<KernelAttr> PriorityReplayBufferCreateGpuKernel::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

bool PriorityReplayBufferCreateGpuKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(handle, handle_device_, sizeof(handle_), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy failed.");
  return true;
}

PriorityReplayBufferPushGpuKernel::~PriorityReplayBufferPushGpuKernel() {
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  if (handle_device_) {
    allocator.FreeTensorMem(handle_device_);
  }
}

bool PriorityReplayBufferPushGpuKernel::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::PriorityReplayBufferPush>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast PriorityReplayBufferPush ops failed!";
    return false;
  }

  handle_ = kernel_ptr->get_handle();
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);

  num_item_ = prioriory_replay_buffer_->schema().size();
  default_priority_ = inputs.size() == num_item_;

  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  handle_device_ = static_cast<int64_t *>(allocator.AllocTensorMem(sizeof(handle_)));
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpy(handle_device_, &handle_, sizeof(handle_), cudaMemcpyHostToDevice),
                                    "cudaMemcpy failed.");

  for (size_t i = 0; i < inputs.size(); i++) {
    TypeId type_id = inputs[i]->GetDtype();
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    const std::vector<int64_t> &shape = inputs[i]->GetShapeVector();
    size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    input_size_list_.push_back(tensor_size);
  }

  output_size_list_.push_back(sizeof(handle_));
  return true;
}

bool PriorityReplayBufferPushGpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  // Return a placeholder in case of dead code eliminate optimization.
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  float *priority = default_priority_ ? nullptr : GetDeviceAddress<float>(inputs, num_item_);

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(handle, handle_device_, sizeof(handle_), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy failed.");

  return prioriory_replay_buffer_->Push(inputs, priority, stream);
}

std::vector<KernelAttr> PriorityReplayBufferPushGpuKernel::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

bool PriorityReplayBufferSampleGpuKernel::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::PriorityReplayBufferSample>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast PriorityReplayBufferSample ops failed!";
    return false;
  }

  handle_ = kernel_ptr->get_handle();
  batch_size_ = kernel_ptr->get_batch_size();
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);

  for (size_t i = 0; i < outputs.size(); i++) {
    TypeId type_id = outputs[i]->GetDtype();
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    const std::vector<int64_t> &shape = outputs[i]->GetShapeVector();
    size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    output_size_list_.push_back(tensor_size);
  }

  return true;
}

bool PriorityReplayBufferSampleGpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto beta = GetDeviceAddress<float>(inputs, 0);
  auto indices = GetDeviceAddress<size_t>(outputs, 0);
  auto weights = GetDeviceAddress<float>(outputs, 1);
  std::vector<AddressPtr> transition;
  for (size_t i = 2; i < outputs.size(); i++) {
    transition.push_back(outputs[i]);
  }

  return prioriory_replay_buffer_->Sample(batch_size_, beta, indices, weights, transition,
                                          reinterpret_cast<cudaStream_t>(stream_ptr));
}

std::vector<KernelAttr> PriorityReplayBufferSampleGpuKernel::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

PriorityReplayBufferUpdateGpuKernel::~PriorityReplayBufferUpdateGpuKernel() {
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  if (handle_device_) {
    allocator.FreeTensorMem(handle_device_);
  }
}

bool PriorityReplayBufferUpdateGpuKernel::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::PriorityReplayBufferUpdate>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast PriorityReplayBufferUpdate ops failed!";
    return false;
  }

  auto indices_shape = inputs[0]->GetShapeVector();
  MS_EXCEPTION_IF_CHECK_FAIL(indices_shape.size() == 1, "The indices rank should be 1.");
  batch_size_ = indices_shape[0];

  handle_ = kernel_ptr->get_handle();
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);

  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  handle_device_ = static_cast<int64_t *>(allocator.AllocTensorMem(sizeof(handle_)));
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpy(handle_device_, &handle_, sizeof(handle_), cudaMemcpyHostToDevice),
                                    "cudaMemcpy failed.");

  for (size_t i = 0; i < inputs.size(); i++) {
    TypeId type_id = inputs[i]->GetDtype();
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    const std::vector<int64_t> &shape = inputs[i]->GetShapeVector();
    size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    input_size_list_.push_back(tensor_size);
  }

  output_size_list_.push_back(sizeof(handle_));
  return true;
}

bool PriorityReplayBufferUpdateGpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto indices = GetDeviceAddress<size_t>(inputs, 0);
  auto priorities = GetDeviceAddress<float>(inputs, 1);
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);

  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(handle, handle_device_, sizeof(handle_), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy failed.");

  return prioriory_replay_buffer_->UpdatePriorities(indices, priorities, batch_size_, stream);
}

std::vector<KernelAttr> PriorityReplayBufferUpdateGpuKernel::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

bool PriorityReplayBufferDestroyGpuKernel::Init(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::PriorityReplayBufferDestroy>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast PriorityReplayBufferDestroy ops failed!";
    return false;
  }

  handle_ = kernel_ptr->get_handle();
  output_size_list_.push_back(sizeof(handle_));
  return true;
}

bool PriorityReplayBufferDestroyGpuKernel::Launch(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &,
                                                  const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  PriorityReplayBufferFactory::GetInstance().Delete(handle_);

  // Apply host to device memory copy since it is not performance critical path.
  auto handle = GetDeviceAddress<float>(outputs, 0);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(handle, &handle_, sizeof(handle_), cudaMemcpyHostToDevice,
                                                    reinterpret_cast<cudaStream_t>(stream_ptr)),
                                    "cudaMemcpy failed.");
  return true;
}

std::vector<KernelAttr> PriorityReplayBufferDestroyGpuKernel::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}
}  // namespace kernel
}  // namespace mindspore
