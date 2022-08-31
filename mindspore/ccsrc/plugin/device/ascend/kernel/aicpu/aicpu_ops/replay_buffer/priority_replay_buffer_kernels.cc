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
constexpr size_t kBetaIndex = 0;
constexpr size_t kIndicesIndex = 1;
constexpr size_t kWeightsIndex = 2;
constexpr size_t kTransitionIndex = 3;
constexpr size_t kUpdateOpInputNum = 2;

uint32_t PriorityReplayBufferCreate::ParseKernelParam() {
  AICPU_LOGI("Enter ParseKernelParam.");

  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  capacity_ = attrs["capacity"].i();
  alpha_ = attrs["alpha"].f();
  int64_t seed1 = attrs["seed"].i();
  int64_t seed2 = attrs["seed2"].i();

  std::random_device rd;
  seed_ = (seed2 != 0) ? seed2 : (seed1 != 0) ? seed1 : rd();

  aicpuops::AttrValue_ArrayValue shape = attrs["schema"].array();
  for (int i = 0; i < shape.i_size(); i++) {
    schema_.push_back(shape.i(i));
  }

  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferCreate::DoCompute() {
  AICPU_LOGI("Do compute start");
  int64_t handle;
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer;
  auto &factory = PriorityReplayBufferFactory::GetInstance();
  std::tie(handle, prioriory_replay_buffer) = factory.Create(seed_, alpha_, capacity_, schema_);

  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[0]);
  output_data[0] = handle;

  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferPush::ParseKernelParam() {
  AICPU_LOGI("Enter ParseKernelParam.");

  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();

  const size_t &num_input = node_def_.inputs_size();
  for (size_t i = 0; i < num_input; i++) {
    ::aicpuops::Tensor input = node_def_.inputs(static_cast<int>(i));
    const auto &dtype = static_cast<::aicpuops::DataType>(input.tensor_type());
    size_t len = GetDataTypeSize(dtype);
    ::aicpuops::TensorShape shape = input.tensor_shape();
    for (int j = 0; j < shape.dim_size(); j++) {
      len *= static_cast<size_t>(shape.dim(j).size());
    }
    (void)inputs_.emplace_back(std::make_shared<Address>(reinterpret_cast<void *>(io_addrs_[i]), len));
  }

  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferPush::DoCompute() {
  AICPU_LOGI("Do compute start");

  auto buffer = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  AICPU_CHECK_NULLPTR(buffer, kAicpuKernelStateFailed, "The instance not exist.");
  (void)buffer->Push(inputs_);

  const size_t &num_input = node_def_.inputs_size();
  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[num_input]);
  output_data[0] = handle_;

  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferSample::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();
  batch_size_ = static_cast<size_t>(attrs["batch_size"].i());

  aicpuops::AttrValue_ArrayValue shape = attrs["schema"].array();
  for (int i = 0; i < shape.i_size(); i++) {
    schema_.push_back(shape.i(i));
  }

  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferSample::DoCompute() {
  std::vector<size_t> indices;
  std::vector<float> weights;
  std::vector<std::vector<AddressPtr>> samples;
  auto prioriory_replay_buffer = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  AICPU_CHECK_NULLPTR(prioriory_replay_buffer, kAicpuKernelStateFailed, "The instance not exist.");
  auto beta = reinterpret_cast<float *>(io_addrs_[kBetaIndex]);
  std::tie(indices, weights, samples) = prioriory_replay_buffer->Sample(batch_size_, beta[0]);

  auto *indices_data = reinterpret_cast<void *>(io_addrs_[kIndicesIndex]);
  auto ret = memcpy_s(indices_data, batch_size_ * sizeof(int64_t), indices.data(), batch_size_ * sizeof(int64_t));
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return kAicpuKernelStateInternalError;
  }

  auto *weights_data = reinterpret_cast<void *>(io_addrs_[kWeightsIndex]);
  ret = memcpy_s(weights_data, batch_size_ * sizeof(float), weights.data(), batch_size_ * sizeof(float));
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return kAicpuKernelStateInternalError;
  }

  for (size_t transition_index = 0; transition_index < samples.size(); transition_index++) {
    const std::vector<AddressPtr> &transition = samples[transition_index];
    for (size_t item_index = 0; item_index < schema_.size(); item_index++) {
      void *offset =
        reinterpret_cast<uint8_t *>(io_addrs_[item_index + kTransitionIndex]) + schema_[item_index] * transition_index;
      ret = memcpy_s(offset, transition[item_index]->size, transition[item_index]->addr, transition[item_index]->size);
      if (ret != EOK) {
        AICPU_LOGE("memcpy_s() failed: %d.", ret);
        return kAicpuKernelStateInternalError;
      }
    }
  }

  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferUpdate::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();

  const size_t &num_input = node_def_.inputs_size();
  if (num_input != kUpdateOpInputNum) {
    AICPU_LOGE("The input num should be 2, get: %d.", num_input);
    return kAicpuKernelStateInternalError;
  }
  for (size_t i = 0; i < num_input; i++) {
    ::aicpuops::Tensor input = node_def_.inputs(static_cast<int>(i));
    const auto &dtype = static_cast<::aicpuops::DataType>(input.tensor_type());
    size_t len = GetDataTypeSize(dtype);
    ::aicpuops::TensorShape shape = input.tensor_shape();
    for (int j = 0; j < shape.dim_size(); j++) {
      len *= static_cast<size_t>(shape.dim(j).size());
    }
    (void)inputs_.emplace_back(std::make_shared<Address>(reinterpret_cast<void *>(io_addrs_[i]), len));
  }

  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferUpdate::DoCompute() {
  batch_size_ = static_cast<size_t>(node_def_.inputs(0).tensor_shape().dim(0).size());
  std::vector<size_t> indices(batch_size_);
  std::vector<float> priorities(batch_size_);

  void *indices_data = reinterpret_cast<void *>(io_addrs_[0]);
  auto ret = memcpy_s(indices.data(), inputs_[0]->size, indices_data, inputs_[0]->size);
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return kAicpuKernelStateInternalError;
  }

  void *priorities_data = reinterpret_cast<void *>(io_addrs_[1]);
  ret = memcpy_s(priorities.data(), inputs_[1]->size, priorities_data, inputs_[1]->size);
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return kAicpuKernelStateInternalError;
  }

  auto prioriory_replay_buffer = PriorityReplayBufferFactory::GetInstance().GetByHandle(handle_);
  AICPU_CHECK_NULLPTR(prioriory_replay_buffer, kAicpuKernelStateFailed, "The instance not exist.");
  (void)prioriory_replay_buffer->UpdatePriorities(indices, priorities);

  // Return a placeholder in case of dead code eliminate optimization.
  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[inputs_.size()]);
  output_data[0] = handle_;
  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferDestroy::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();
  return kAicpuKernelStateSucess;
}

uint32_t PriorityReplayBufferDestroy::DoCompute() {
  PriorityReplayBufferFactory::GetInstance().Delete(handle_);
  return kAicpuKernelStateSucess;
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

__attribute__((visibility("default"))) uint32_t PriorityReplayBufferDestroy(void *param) {
  aicpu::PriorityReplayBufferDestroy prb_destroy;
  return prb_destroy.Compute(param);
}
}
