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

#include "replay_buffer/priority_replay_buffer_kernels.h"

#include <string>
#include <memory>
#include <vector>
#include "replay_buffer/replay_buffer_factory.h"

namespace aicpu {
using PriorityReplayBufferFactory = ReplayBufferFactory<PriorityReplayBuffer>;
constexpr size_t kIndicesIndex = 0;
constexpr size_t kInWeightsIndex = 1;
constexpr size_t kTransitionIndex = 2;
constexpr size_t kUpdateOpInputNum = 2;

uint32_t PriorityReplayBufferCreate::ParseKernelParam() {
  AICPU_LOGI("Enter ParseKernelParam.");

  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  capacity_ = attrs["capacity"].i();
  alpha_ = attrs["alpha"].f();
  beta_ = attrs["beta"].f();
  int64_t seed1 = attrs["seed"].i();
  int64_t seed2 = attrs["seed2"].i();

  std::random_device rd;
  seed_ = (seed2 != 0) ? seed2 : (seed1 != 0) ? seed1 : rd();

  aicpuops::AttrValue_ArrayValue shape = attrs["schema"].array();
  for (int i = 0; i < shape.i_size(); i++) {
    schema_.push_back(shape.i(i));
  }

  return AICPU_KERNEL_STATE_SUCCESS;
}

uint32_t PriorityReplayBufferCreate::DoCompute() {
  AICPU_LOGI("Do compute start");
  int64_t handle;
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer;
  auto &factory = PriorityReplayBufferFactory::GetInstance();
  std::tie(handle, prioriory_replay_buffer) = factory.Create(seed_, alpha_, beta_, capacity_, schema_);

  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[0]);
  output_data[0] = handle;

  return AICPU_KERNEL_STATE_SUCCESS;
}

uint32_t PriorityReplayBufferPush::ParseKernelParam() {
  AICPU_LOGI("Enter ParseKernelParam.");

  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();

  const size_t &num_input = node_def_.inputs_size();
  for (size_t i = 0; i < num_input; i++) {
    ::aicpuops::Tensor input = node_def_.inputs(i);
    const auto &dtype = static_cast<::aicpuops::DataType>(input.tensor_type());
    size_t len = GetDataTypeSize(dtype);
    ::aicpuops::TensorShape shape = input.tensor_shape();
    for (int j = 0; j < shape.dim_size(); j++) {
      len *= shape.dim(j).size();
    }
    inputs_.emplace_back(std::make_shared<Address>(reinterpret_cast<void *>(io_addrs_[i]), len));
  }

  return AICPU_KERNEL_STATE_SUCCESS;
}

uint32_t PriorityReplayBufferPush::DoCompute() {
  AICPU_LOGI("Do compute start");

  auto buffer = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  buffer->Push(inputs_);

  const size_t &num_input = node_def_.inputs_size();
  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[num_input]);
  output_data[0] = handle_;

  return AICPU_KERNEL_STATE_SUCCESS;
}

uint32_t PriorityReplayBufferSample::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();
  batch_size_ = attrs["batch_size"].i();

  aicpuops::AttrValue_ArrayValue shape = attrs["schema"].array();
  for (int i = 0; i < shape.i_size(); i++) {
    schema_.push_back(shape.i(i));
  }

  return AICPU_KERNEL_STATE_SUCCESS;
}

uint32_t PriorityReplayBufferSample::DoCompute() {
  std::vector<size_t> indices;
  std::vector<float> weights;
  std::vector<std::vector<AddressPtr>> samples;
  auto prioriory_replay_buffer = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  std::tie(indices, weights, samples) = prioriory_replay_buffer->Sample(batch_size_);

  auto *indices_data = reinterpret_cast<void *>(io_addrs_[0]);
  auto ret = memcpy_s(indices_data, batch_size_ * sizeof(int64_t), indices.data(), batch_size_ * sizeof(int64_t));
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return AICPU_KERNEL_STATE_INTERNAL_ERROR;
  }

  auto *weights_data = reinterpret_cast<void *>(io_addrs_[1]);
  ret = memcpy_s(weights_data, batch_size_ * sizeof(float), weights.data(), batch_size_ * sizeof(float));
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return AICPU_KERNEL_STATE_INTERNAL_ERROR;
  }

  for (size_t transition_index = 0; transition_index < samples.size(); transition_index++) {
    const std::vector<AddressPtr> &transition = samples[transition_index];
    for (size_t item_index = 0; item_index < schema_.size(); item_index++) {
      void *offset =
        reinterpret_cast<uint8_t *>(io_addrs_[item_index + kTransitionIndex]) + schema_[item_index] * transition_index;
      ret = memcpy_s(offset, transition[item_index]->size, transition[item_index]->addr, transition[item_index]->size);
      if (ret != EOK) {
        AICPU_LOGE("memcpy_s() failed: %d.", ret);
        return AICPU_KERNEL_STATE_INTERNAL_ERROR;
      }
    }
  }

  return AICPU_KERNEL_STATE_SUCCESS;
}

uint32_t PriorityReplayBufferUpdate::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();

  const size_t &num_input = node_def_.inputs_size();
  if (num_input != kUpdateOpInputNum) {
    AICPU_LOGE("The input num should be 2, get: %d.", num_input);
    return AICPU_KERNEL_STATE_INTERNAL_ERROR;
  }
  for (size_t i = 0; i < num_input; i++) {
    ::aicpuops::Tensor input = node_def_.inputs(i);
    const auto &dtype = static_cast<::aicpuops::DataType>(input.tensor_type());
    size_t len = GetDataTypeSize(dtype);
    ::aicpuops::TensorShape shape = input.tensor_shape();
    for (int j = 0; j < shape.dim_size(); j++) {
      len *= shape.dim(j).size();
    }
    inputs_.emplace_back(std::make_shared<Address>(reinterpret_cast<void *>(io_addrs_[i]), len));
  }

  return AICPU_KERNEL_STATE_SUCCESS;
}

uint32_t PriorityReplayBufferUpdate::DoCompute() {
  batch_size_ = node_def_.inputs(0).tensor_shape().dim(0).size();
  std::vector<size_t> indices(batch_size_);
  std::vector<float> priorities(batch_size_);

  void *indices_data = reinterpret_cast<void *>(io_addrs_[0]);
  auto ret = memcpy_s(indices.data(), inputs_[0]->size, indices_data, inputs_[0]->size);
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return AICPU_KERNEL_STATE_INTERNAL_ERROR;
  }

  void *priorities_data = reinterpret_cast<void *>(io_addrs_[1]);
  ret = memcpy_s(priorities.data(), inputs_[1]->size, priorities_data, inputs_[1]->size);
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return AICPU_KERNEL_STATE_INTERNAL_ERROR;
  }

  auto prioriory_replay_buffer = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  prioriory_replay_buffer->UpdatePriorities(indices, priorities);

  // Return a placeholder in case of dead code eliminate optimization.
  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[inputs_.size()]);
  output_data[0] = handle_;
  return AICPU_KERNEL_STATE_SUCCESS;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t PriorityReplayBufferCreate(void *param) {
  aicpu::PriorityReplayBufferCreate prb_create;
  return prb_create.Compute(param);
}

__attribute__((visibility("default"))) uint32_t PriorityReplayBufferPush(void *param) {
  aicpu::PriorityReplayBufferPush prb_push;
  return prb_push.Compute(param);
}

__attribute__((visibility("default"))) uint32_t PriorityReplayBufferSample(void *param) {
  aicpu::PriorityReplayBufferSample prb_sample;
  return prb_sample.Compute(param);
}

__attribute__((visibility("default"))) uint32_t PriorityReplayBufferUpdate(void *param) {
  aicpu::PriorityReplayBufferUpdate prb_update;
  return prb_update.Compute(param);
}
}
