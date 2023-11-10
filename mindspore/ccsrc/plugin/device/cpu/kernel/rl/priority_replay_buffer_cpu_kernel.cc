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

#include "plugin/device/cpu/kernel/rl/priority_replay_buffer_cpu_kernel.h"

#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include "kernel/kernel.h"
#include "plugin/factory/replay_buffer_factory.h"

namespace mindspore {
namespace kernel {
using PriorityReplayBufferFactory = ReplayBufferFactory<PriorityReplayBuffer>;
constexpr size_t kIndicesIndex = 0;
constexpr size_t kInWeightsIndex = 1;
constexpr size_t kTransitionIndex = 2;

int PriorityReplayBufferCreateCpuKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  const int64_t &capacity = GetValue<int64_t>(primitive_->GetAttr("capacity"));
  const float &alpha = GetValue<float>(primitive_->GetAttr("alpha"));
  const auto &dtypes = GetValue<std::vector<TypePtr>>(primitive_->GetAttr("dtypes"));
  const auto &shapes = GetValue<std::vector<std::vector<int64_t>>>(primitive_->GetAttr("shapes"));
  const int64_t &seed0 = GetValue<int64_t>(primitive_->GetAttr("seed0"));
  const int64_t &seed1 = GetValue<int64_t>(primitive_->GetAttr("seed1"));

  MS_EXCEPTION_IF_CHECK_FAIL(dtypes.size() == shapes.size(), "The dtype and shapes must be the same.");
  std::vector<size_t> schema;
  for (size_t i = 0; i < shapes.size(); i++) {
    size_t num_element = std::accumulate(shapes[i].begin(), shapes[i].end(), 1ULL, std::multiplies<size_t>());
    size_t type_size = GetTypeByte(dtypes[i]);
    schema.push_back(num_element * type_size);
  }

  unsigned int seed = 0;
  std::random_device rd;
  if (seed1 != 0) {
    seed = static_cast<unsigned int>(seed1);
  } else if (seed0 != 0) {
    seed = static_cast<unsigned int>(seed0);
  } else {
    seed = rd();
  }

  auto &factory = PriorityReplayBufferFactory::GetInstance();
  std::tie(handle_, prioriory_replay_buffer_) = factory.Create(seed, alpha, capacity, schema);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);
  return KRET_OK;
}

bool PriorityReplayBufferCreateCpuKernel::Launch(const std::vector<KernelTensor *> &,
                                                 const std::vector<KernelTensor *> &,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  *handle = handle_;
  return true;
}

int PriorityReplayBufferPushCpuKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  handle_ = GetValue<int64_t>(primitive_->GetAttr("handle"));
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);
  return KRET_OK;
}

bool PriorityReplayBufferPushCpuKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs) {
  std::vector<AddressPtr> inputs_addr;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input_addr = std::make_shared<Address>(inputs[i]->device_ptr(), inputs[i]->size());
    inputs_addr.push_back(input_addr);
  }
  (void)prioriory_replay_buffer_->Push(inputs_addr);

  // Return a placeholder in case of dead code eliminate optimization.
  auto handle = GetDeviceAddress<int64_t>(outputs, kIndex0);
  *handle = handle_;
  return true;
}

int PriorityReplayBufferSampleCpuKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  handle_ = GetValue<int64_t>(primitive_->GetAttr("handle"));
  batch_size_ = LongToSize(GetValue<int64_t>(primitive_->GetAttr("batch_size")));
  const auto &dtypes = GetValue<std::vector<TypePtr>>(primitive_->GetAttr("dtypes"));
  const auto &shapes = GetValue<std::vector<std::vector<int64_t>>>(primitive_->GetAttr("shapes"));
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);

  for (size_t i = 0; i < shapes.size(); i++) {
    size_t num_element = std::accumulate(shapes[i].begin(), shapes[i].end(), 1ULL, std::multiplies<size_t>());
    size_t type_size = GetTypeByte(dtypes[i]);
    schema_.push_back(num_element * type_size);
  }
  return KRET_OK;
}

bool PriorityReplayBufferSampleCpuKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &,
                                                 const std::vector<KernelTensor *> &outputs) {
  std::vector<size_t> indices;
  std::vector<float> weights;
  std::vector<std::vector<AddressPtr>> samples;

  auto beta = reinterpret_cast<float *>(inputs[kIndex0]->device_ptr());
  std::tie(indices, weights, samples) = prioriory_replay_buffer_->Sample(batch_size_, beta[0]);

  MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == schema_.size() + kTransitionIndex,
                             "The dtype and shapes must be the same.");
  MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(outputs[kIndicesIndex]->device_ptr(), outputs[kIndicesIndex]->size(),
                                      indices.data(), batch_size_ * sizeof(int64_t)) == EOK,
                             "memcpy_s() failed.");
  MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(outputs[kInWeightsIndex]->device_ptr(), outputs[kInWeightsIndex]->size(),
                                      weights.data(), batch_size_ * sizeof(float)) == EOK,
                             "memcpy_s() failed.");

  for (size_t transition_index = 0; transition_index < samples.size(); transition_index++) {
    const std::vector<AddressPtr> &transition = samples[transition_index];
    for (size_t item_index = 0; item_index < schema_.size(); item_index++) {
      void *offset = reinterpret_cast<uint8_t *>(outputs[item_index + kTransitionIndex]->device_ptr()) +
                     schema_[item_index] * transition_index;
      MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(offset, outputs[item_index + kTransitionIndex]->size(),
                                          transition[item_index]->addr, transition[item_index]->size) == EOK,
                                 "memcpy_s() failed.");
    }
  }

  return true;
}

int PriorityReplayBufferUpdateCpuKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  indices_shape_ = inputs[kIndex0]->GetShapeVector();
  priorities_shape_ = inputs[kIndex1]->GetShapeVector();
  MS_EXCEPTION_IF_CHECK_FAIL(indices_shape_.size() != 0, "The indices shape can not be null.");
  MS_EXCEPTION_IF_CHECK_FAIL(priorities_shape_.size() != 0, "The priorities shape can not be null.");

  handle_ = GetValue<int64_t>(primitive_->GetAttr("handle"));
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);
  return KRET_OK;
}

bool PriorityReplayBufferUpdateCpuKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &,
                                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == 2, "inputs must be 2.");
  std::vector<size_t> indices(indices_shape_[kIndex0]);
  std::vector<float> priorities(priorities_shape_[kIndex0]);
  MS_EXCEPTION_IF_CHECK_FAIL(
    memcpy_s(indices.data(), inputs[kIndex0]->size(), inputs[kIndex0]->device_ptr(), inputs[kIndex0]->size()) == EOK,
    "memcpy_s() failed.");
  MS_EXCEPTION_IF_CHECK_FAIL(
    memcpy_s(priorities.data(), inputs[kIndex1]->size(), inputs[kIndex1]->device_ptr(), inputs[kIndex1]->size()) == EOK,
    "memcpy_s() failed.");
  (void)prioriory_replay_buffer_->UpdatePriorities(indices, priorities);

  // Return a placeholder in case of dead code eliminate optimization.
  auto handle = GetDeviceAddress<int64_t>(outputs, kIndex0);
  *handle = handle_;
  return true;
}

int PriorityReplayBufferDestroyCpuKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  handle_ = GetValue<int64_t>(primitive_->GetAttr("handle"));
  return KRET_OK;
}

bool PriorityReplayBufferDestroyCpuKernel::Launch(const std::vector<KernelTensor *> &,
                                                  const std::vector<KernelTensor *> &,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto &factory = PriorityReplayBufferFactory::GetInstance();
  factory.Delete(handle_);

  auto handle = GetDeviceAddress<int64_t>(outputs, kIndex0);
  *handle = handle_;
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PriorityReplayBufferCreate, PriorityReplayBufferCreateCpuKernel);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PriorityReplayBufferPush, PriorityReplayBufferPushCpuKernel);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PriorityReplayBufferSample, PriorityReplayBufferSampleCpuKernel);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PriorityReplayBufferUpdate, PriorityReplayBufferUpdateCpuKernel);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PriorityReplayBufferDestroy, PriorityReplayBufferDestroyCpuKernel);
}  // namespace kernel
}  // namespace mindspore
