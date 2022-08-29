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

#include "replay_buffer/reservoir_replay_buffer_kernels.h"

#include <string>
#include <memory>
#include <vector>
#include "replay_buffer/replay_buffer_factory.h"

namespace aicpu {
using ReservoirReplayBufferFactory = ReplayBufferFactory<ReservoirReplayBuffer>;
uint32_t ReservoirReplayBufferCreate::ParseKernelParam() {
  AICPU_LOGI("Enter ParseKernelParam.");

  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  int64_t seed1 = attrs["seed"].i();
  int64_t seed2 = attrs["seed2"].i();
  std::random_device rd;
  seed_ = (seed2 != 0) ? seed2 : (seed1 != 0) ? seed1 : rd();

  capacity_ = attrs["capacity"].i();

  aicpuops::AttrValue_ArrayValue shape = attrs["schema"].array();
  for (int i = 0; i < shape.i_size(); i++) {
    schema_.push_back(shape.i(i));
  }

  return kAicpuKernelStateSucess;
}

uint32_t ReservoirReplayBufferCreate::DoCompute() {
  AICPU_LOGI("Do compute start");
  int64_t handle;
  std::shared_ptr<ReservoirReplayBuffer> reservoir_replay_buffer;
  auto &factory = ReservoirReplayBufferFactory::GetInstance();
  std::tie(handle, reservoir_replay_buffer) = factory.Create(seed_, capacity_, schema_);

  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[0]);
  output_data[0] = handle;

  return kAicpuKernelStateSucess;
}

uint32_t ReservoirReplayBufferPush::ParseKernelParam() {
  AICPU_LOGI("Enter ParseKernelParam.");

  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();

  const size_t &num_input = node_def_.inputs_size();
  for (size_t i = 0; i < num_input; i++) {
    ::aicpuops::Tensor input = node_def_.inputs(SizeToInt(i));
    const auto &dtype = static_cast<::aicpuops::DataType>(input.tensor_type());
    size_t len = GetDataTypeSize(dtype);
    ::aicpuops::TensorShape shape = input.tensor_shape();
    for (int j = 0; j < shape.dim_size(); j++) {
      len *= LongToSize(shape.dim(j).size());
    }
    (void)inputs_.emplace_back(std::make_shared<Address>(reinterpret_cast<void *>(io_addrs_[i]), len));
  }

  return kAicpuKernelStateSucess;
}

uint32_t ReservoirReplayBufferPush::DoCompute() {
  AICPU_LOGI("Do compute start");

  auto buffer = ReservoirReplayBufferFactory::GetInstance().GetByHandle(handle_);
  (void)buffer->Push(inputs_);

  const size_t &num_input = node_def_.inputs_size();
  auto *output_data = reinterpret_cast<int64_t *>(io_addrs_[num_input]);
  output_data[0] = handle_;

  return kAicpuKernelStateSucess;
}

uint32_t ReservoirReplayBufferSample::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();
  batch_size_ = attrs["batch_size"].i();

  aicpuops::AttrValue_ArrayValue shape = attrs["schema"].array();
  for (int i = 0; i < shape.i_size(); i++) {
    schema_.push_back(shape.i(i));
  }

  return kAicpuKernelStateSucess;
}

uint32_t ReservoirReplayBufferSample::DoCompute() {
  auto reservoir_replay_buffer = ReservoirReplayBufferFactory::GetInstance().GetByHandle(handle_);
  return reservoir_replay_buffer->Sample(batch_size_, io_addrs_);
}

uint32_t ReservoirReplayBufferDestroy::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  handle_ = attrs["handle"].i();
  return kAicpuKernelStateSucess;
}

uint32_t ReservoirReplayBufferDestroy::DoCompute() {
  ReservoirReplayBufferFactory::GetInstance().Delete(handle_);
  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t ReservoirReplayBufferCreate(void *param) {
  aicpu::ReservoirReplayBufferCreate prb_create;
  return prb_create.Compute(param);
}

__attribute__((visibility("default"))) uint32_t ReservoirReplayBufferPush(void *param) {
  aicpu::ReservoirReplayBufferPush prb_push;
  return prb_push.Compute(param);
}

__attribute__((visibility("default"))) uint32_t ReservoirReplayBufferSample(void *param) {
  aicpu::ReservoirReplayBufferSample prb_sample;
  return prb_sample.Compute(param);
}

__attribute__((visibility("default"))) uint32_t ReservoirReplayBufferDestroy(void *param) {
  aicpu::ReservoirReplayBufferDestroy prb_destroy;
  return prb_destroy.Compute(param);
}
}
