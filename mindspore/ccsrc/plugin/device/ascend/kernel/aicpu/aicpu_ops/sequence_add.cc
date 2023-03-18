/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/sequence_add.h"
#include <string>
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
constexpr auto kSequenceAddInputNum = 2;
constexpr auto kSequenceAddOutputNum = 1;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;

uint32_t SequenceAddKernel::ParseKernelParam() {
  if (node_def_.inputs_size() != kSequenceAddInputNum) {
    AICPU_LOGE("For 'SequenceAdd', input number must be 2, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }

  if (node_def_.outputs_size() != kSequenceAddOutputNum) {
    AICPU_LOGE("For 'SequenceAdd', output number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }

  aicpuops::Tensor input_0_tensor = node_def_.inputs(kDim0);
  input_0_data_type_ = static_cast<aicpuops::DataType>(input_0_tensor.tensor_type());
  input_0_data_size_ = GetTensorMemSizeByShape(input_0_tensor);

  input_1_data_size_ = GetTensorMemSizeByShape(node_def_.inputs(kDim1));
  output_data_size_ = GetTensorMemSizeByShape(node_def_.outputs(kDim0));
  return kAicpuKernelStateSucess;
}

template <typename T>
uint32_t SequenceAddKernel::SequenceAddTask() {
  const auto input_0_addr = reinterpret_cast<void *>(io_addrs_[kDim0]);
  const auto input_1_addr = reinterpret_cast<void *>(io_addrs_[kDim1]);
  auto output_addr = reinterpret_cast<void *>(io_addrs_[kDim2]);
  if (input_0_data_size_ + input_1_data_size_ > output_data_size_) {
    AICPU_LOGE(
      "For 'SequenceAdd', the size of 'input_0 + input_1': {%d + %d} is not equal to the size of output:{ % d } ",
      input_0_data_size_, input_1_data_size_, output_data_size_);
    return kAicpuKernelStateInvalid;
  }

  auto cp_ret = memcpy_s(output_addr, output_data_size_, input_0_addr, input_0_data_size_);
  if (cp_ret != EOK) {
    AICPU_LOGE("For 'SequenceAdd',  memcpy for input 0 error, errorno: %d, size: %d.", cp_ret, input_0_data_size_);
    return kAicpuKernelStateInvalid;
  }

  output_addr = reinterpret_cast<int8_t *>(output_addr) + input_0_data_size_;
  cp_ret = memcpy_s(output_addr, output_data_size_ - input_0_data_size_, input_1_addr, input_1_data_size_);
  if (cp_ret != EOK) {
    AICPU_LOGE("For 'SequenceAdd',  memcpy for input 1 error, errorno: %d, size: %d.", cp_ret, input_1_data_size_);
    return kAicpuKernelStateInvalid;
  }

  return kAicpuKernelStateSucess;
}

uint32_t SequenceAddKernel::DoCompute() {
  switch (input_0_data_type_) {
    case aicpuops::DataType::MS_INT32:
      return SequenceAddTask<int>();
    case aicpuops::DataType::MS_INT64:
      return SequenceAddTask<int64_t>();
    case aicpuops::DataType::MS_FLOAT32:
      return SequenceAddTask<float>();
    case aicpuops::DataType::MS_FLOAT64:
      return SequenceAddTask<double>();
    default:
      AICPU_LOGE("SequenceAdd kernel data type [%s] not support.", static_cast<int>(input_0_data_type_));
      return kAicpuKernelStateInvalid;
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t SequenceAdd(void *param) {
  aicpu::SequenceAddKernel sequence_add_kernel;
  return sequence_add_kernel.Compute(param);
}
}
