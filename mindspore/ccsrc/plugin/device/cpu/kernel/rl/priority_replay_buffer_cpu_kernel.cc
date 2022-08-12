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

void PriorityReplayBufferCreateCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  const int64_t &capacity = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "capacity");
  const float &alpha = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "alpha");
  const auto &dtypes = common::AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "dtypes");
  const auto &shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "shapes");
  const int64_t &seed0 = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed0");
  const int64_t &seed1 = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed1");

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
}

bool PriorityReplayBufferCreateCpuKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs) {
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  *handle = handle_;
  return true;
}

void PriorityReplayBufferPushCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  handle_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "handle");
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);
}

bool PriorityReplayBufferPushCpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs) {
  (void)prioriory_replay_buffer_->Push(inputs);

  // Return a placeholder in case of dead code eliminate optimization.
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  *handle = handle_;
  return true;
}

void PriorityReplayBufferSampleCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  handle_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "handle");
  batch_size_ = LongToSize(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "batch_size"));
  const auto &dtypes = common::AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "dtypes");
  const auto &shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "shapes");
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);

  for (size_t i = 0; i < shapes.size(); i++) {
    size_t num_element = std::accumulate(shapes[i].begin(), shapes[i].end(), 1ULL, std::multiplies<size_t>());
    size_t type_size = GetTypeByte(dtypes[i]);
    schema_.push_back(num_element * type_size);
  }
}

bool PriorityReplayBufferSampleCpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs) {
  std::vector<size_t> indices;
  std::vector<float> weights;
  std::vector<std::vector<AddressPtr>> samples;

  auto beta = reinterpret_cast<float *>(inputs[0]->addr);
  std::tie(indices, weights, samples) = prioriory_replay_buffer_->Sample(batch_size_, beta[0]);

  MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == schema_.size() + kTransitionIndex,
                             "The dtype and shapes must be the same.");
  MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(outputs[kIndicesIndex]->addr, outputs[kIndicesIndex]->size, indices.data(),
                                      batch_size_ * sizeof(int64_t)) == EOK,
                             "memcpy_s() failed.");
  MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(outputs[kInWeightsIndex]->addr, outputs[kInWeightsIndex]->size, weights.data(),
                                      batch_size_ * sizeof(float)) == EOK,
                             "memcpy_s() failed.");

  for (size_t transition_index = 0; transition_index < samples.size(); transition_index++) {
    const std::vector<AddressPtr> &transition = samples[transition_index];
    for (size_t item_index = 0; item_index < schema_.size(); item_index++) {
      void *offset = reinterpret_cast<uint8_t *>(outputs[item_index + kTransitionIndex]->addr) +
                     schema_[item_index] * transition_index;
      MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(offset, outputs[item_index + kTransitionIndex]->size,
                                          transition[item_index]->addr, transition[item_index]->size) == EOK,
                                 "memcpy_s() failed.");
    }
  }

  return true;
}

void PriorityReplayBufferUpdateCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  indices_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  priorities_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  MS_EXCEPTION_IF_CHECK_FAIL(indices_shape_.size() != 0, "The indices shape can not be null.");
  MS_EXCEPTION_IF_CHECK_FAIL(priorities_shape_.size() != 0, "The priorities shape can not be null.");

  handle_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "handle");
  prioriory_replay_buffer_ = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  MS_EXCEPTION_IF_NULL(prioriory_replay_buffer_);
}

bool PriorityReplayBufferUpdateCpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == 2, "inputs must be 2.");
  std::vector<size_t> indices(indices_shape_[0]);
  std::vector<float> priorities(priorities_shape_[0]);
  MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(indices.data(), inputs[0]->size, inputs[0]->addr, inputs[0]->size) == EOK,
                             "memcpy_s() failed.");
  MS_EXCEPTION_IF_CHECK_FAIL(memcpy_s(priorities.data(), inputs[1]->size, inputs[1]->addr, inputs[1]->size) == EOK,
                             "memcpy_s() failed.");
  (void)prioriory_replay_buffer_->UpdatePriorities(indices, priorities);

  // Return a placeholder in case of dead code eliminate optimization.
  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
  *handle = handle_;
  return true;
}

void PriorityReplayBufferDestroyCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  handle_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "handle");
}

bool PriorityReplayBufferDestroyCpuKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                  const std::vector<AddressPtr> &outputs) {
  auto &factory = PriorityReplayBufferFactory::GetInstance();
  factory.Delete(handle_);

  auto handle = GetDeviceAddress<int64_t>(outputs, 0);
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
