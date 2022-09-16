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

#include "plugin/device/cpu/kernel/rl/reservoir_replay_buffer_cpu_kernel.h"

#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include "kernel/kernel.h"
#include "plugin/factory/replay_buffer_factory.h"
#include "mindspore/core/ops/reservoir_replay_buffer.h"

namespace mindspore {
namespace kernel {
using ReservoirReplayBufferFactory = ReplayBufferFactory<ReservoirReplayBuffer>;
constexpr size_t kIndicesIndex = 0;
constexpr size_t kInWeightsIndex = 1;
constexpr size_t kTransitionIndex = 2;

bool ReservoirReplayBufferCreateCpuKernel::Init(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &,
                                                const std::vector<KernelTensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ReservoirReplayBufferCreate>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ReservoirReplayBufferCreate ops failed!";
    return false;
  }

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
  (void)std::transform(schema.begin(), schema.end(), std::back_inserter(schema_in_size),
                       [](const int64_t &arg) -> size_t { return LongToSize(arg); });

  auto &factory = ReservoirReplayBufferFactory::GetInstance();
  const int64_t &capacity = kernel_ptr->get_capacity();
  std::tie(handle_, reservoir_replay_buffer_) = factory.Create(seed, capacity, schema_in_size);
  MS_EXCEPTION_IF_NULL(reservoir_replay_buffer_);

  output_size_list_.push_back(sizeof(handle_));
  return true;
}

bool ReservoirReplayBufferCreateCpuKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                  const std::vector<AddressPtr> &outputs) {
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  *handle = handle_;
  return true;
}

bool ReservoirReplayBufferPushCpuKernel::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ReservoirReplayBufferPush>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ReservoirReplayBufferPush ops failed!";
    return false;
  }

  handle_ = kernel_ptr->get_handle();
  reservoir_replay_buffer_ = ReservoirReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(reservoir_replay_buffer_);

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

bool ReservoirReplayBufferPushCpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                const std::vector<AddressPtr> &outputs) {
  (void)reservoir_replay_buffer_->Push(inputs);

  // Return a placeholder in case of dead code eliminate optimization.
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  *handle = handle_;
  return true;
}

bool ReservoirReplayBufferSampleCpuKernel::Init(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &,
                                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ReservoirReplayBufferSample>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ReservoirReplayBufferSample ops failed!";
    return false;
  }

  handle_ = kernel_ptr->get_handle();
  batch_size_ = LongToSize(kernel_ptr->get_batch_size());
  reservoir_replay_buffer_ = ReservoirReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(reservoir_replay_buffer_);

  for (size_t i = 0; i < outputs.size(); i++) {
    TypeId type_id = outputs[i]->GetDtype();
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    const std::vector<int64_t> &shape = outputs[i]->GetShapeVector();
    size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    output_size_list_.push_back(tensor_size);
  }

  return true;
}

bool ReservoirReplayBufferSampleCpuKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                  const std::vector<AddressPtr> &outputs) {
  return reservoir_replay_buffer_->Sample(batch_size_, outputs);
}

bool ReservoirReplayBufferDestroyCpuKernel::Init(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &,
                                                 const std::vector<KernelTensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ReservoirReplayBufferDestroy>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ReservoirReplayBufferDestroy ops failed!";
    return false;
  }

  handle_ = kernel_ptr->get_handle();
  output_size_list_.push_back(sizeof(handle_));
  return true;
}

bool ReservoirReplayBufferDestroyCpuKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                   const std::vector<AddressPtr> &outputs) {
  auto &factory = ReservoirReplayBufferFactory::GetInstance();
  factory.Delete(handle_);

  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  *handle = handle_;
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ReservoirReplayBufferCreate, ReservoirReplayBufferCreateCpuKernel);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ReservoirReplayBufferPush, ReservoirReplayBufferPushCpuKernel);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ReservoirReplayBufferSample, ReservoirReplayBufferSampleCpuKernel);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ReservoirReplayBufferDestroy, ReservoirReplayBufferDestroyCpuKernel);
}  // namespace kernel
}  // namespace mindspore
