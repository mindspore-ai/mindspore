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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/sequence_add_offset.h"
#include <string>
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
constexpr size_t kSequenceAddOffsetInputNum = 2;
constexpr size_t kSequenceAddOffsetOutputNum = 1;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;

uint32_t SequenceAddOffsetKernel::ParseKernelParam() {
  if (node_def_.inputs_size() != kSequenceAddOffsetInputNum) {
    AICPU_LOGE("For 'SequenceAddOffset', input number must be 2, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }

  if (node_def_.outputs_size() != kSequenceAddOffsetOutputNum) {
    AICPU_LOGE("For 'SequenceAddOffset', output number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }

  aicpuops::Tensor input_0_tensor = node_def_.inputs(kDim0);
  input_0_data_type_ = static_cast<aicpuops::DataType>(input_0_tensor.tensor_type());
  input_0_data_size_ = GetTensorMemSizeByShape(input_0_tensor);

  aicpuops::Tensor input_1_tensor = node_def_.inputs(kDim1);
  auto input_1_data_type = static_cast<aicpuops::DataType>(input_1_tensor.tensor_type());
  if (input_0_data_type_ != input_1_data_type) {
    AICPU_LOGE("For 'SequenceAddOffset', inputs data type must be same, but got %d and %d", input_0_data_type_,
               input_1_data_type);
    return kAicpuKernelStateInvalid;
  }

  aicpuops::Tensor output_tensor = node_def_.outputs(kDim0);
  auto output_data_type = static_cast<aicpuops::DataType>(output_tensor.tensor_type());
  if (output_data_type != aicpuops::DataType::MS_INT64) {
    AICPU_LOGE("For 'SequenceAddOffset', output data type must be int64, but got %d", output_data_type);
    return kAicpuKernelStateInvalid;
  }
  return kAicpuKernelStateSucess;
}

template <typename T>
uint32_t SequenceAddOffsetKernel::SequenceAddOffsetTask() {
  auto output_addr = reinterpret_cast<int64_t *>(io_addrs_[kDim2]);
  output_addr[0] = 0;
  output_addr[1] = input_0_data_size_ / sizeof(T);
  return kAicpuKernelStateSucess;
}

uint32_t SequenceAddOffsetKernel::DoCompute() {
  switch (input_0_data_type_) {
    case aicpuops::DataType::MS_INT32:
      return SequenceAddOffsetTask<int>();
    case aicpuops::DataType::MS_INT64:
      return SequenceAddOffsetTask<int64_t>();
    case aicpuops::DataType::MS_FLOAT32:
      return SequenceAddOffsetTask<float>();
    case aicpuops::DataType::MS_FLOAT64:
      return SequenceAddOffsetTask<double>();
    default:
      AICPU_LOGE("SequenceAddOffset kernel data type [%s] not support.", input_0_data_type_);
      return kAicpuKernelStateInvalid;
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t SequenceAddOffset(void *param) {
  aicpu::SequenceAddOffsetKernel sequence_add_offset_kernel;
  return sequence_add_offset_kernel.Compute(param);
}
}
