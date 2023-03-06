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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/sequence_addn.h"
#include <string>
#include <thread>
#include <complex>
#include "proto/aicpu_tensor.pb.h"
#include "common/atomic_op.h"
#include "utils/eigen_tensor.h"
#include "aicpu_sharder/aicpu_sharder.h"

namespace aicpu {
namespace {
std::vector<int64_t> GetShape(const ::aicpuops::TensorShape &shape) {
  std::vector<int64_t> res;
  for (int i = 0; i < shape.dim_size(); ++i) {
    res.push_back(shape.dim(i).size());
  }
  return res;
}
}  // namespace
constexpr size_t kSequenceAddNInputNum = 1;
constexpr size_t kSequenceAddNOutputNum = 1;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;

uint32_t SequenceAddNKernel::ParseKernelParam() {
  if (node_def_.inputs_size() != kSequenceAddNInputNum) {
    AICPU_LOGE("For 'SequenceAddN', input number must be 1, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }

  if (node_def_.outputs_size() != kSequenceAddNOutputNum) {
    AICPU_LOGE("For 'SequenceAddN', output number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }
  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  input_data_type_ = static_cast<aicpuops::DataType>(input_tensor.tensor_type());
  auto input_shape = GetShape(input_tensor.tensor_shape());
  input_shapes_.push_back(input_shape);
  input_data_size_ = GetTensorMemSizeByShape(node_def_.inputs(kDim0));
  output_data_size_ = GetTensorMemSizeByShape(node_def_.outputs(kDim0));
  return kAicpuKernelStateSucess;
}

template <typename T>
uint32_t SequenceAddNKernel::SequenceAddNTask() {
  const auto inputs_addr = reinterpret_cast<T *>(io_addrs_[kDim0]);
  auto output_addr = reinterpret_cast<T *>(io_addrs_[kDim1]);
  auto element_num = LongToSize(input_shapes_[0][0]);
  auto element_size = output_data_size_ / sizeof(T);
  auto cp_ret = memset_s(output_addr, output_data_size_, 0x0, output_data_size_);
  if (cp_ret != EOK) {
    AICPU_LOGE("For 'SequenceAddN',  memset for output error, errorno: %d, size: %d.", cp_ret, output_data_size_);
    return kAicpuKernelStateInvalid;
  }
  auto input_x_addr = inputs_addr;
  auto sequence_add_n = [this, &output_addr, &input_x_addr](size_t start, size_t end) {
    for (size_t id = start; id < end; id++) {
      AtomicAdd<T>(output_addr + id, input_x_addr[id]);
    }
  };
  const int64_t per_unit_size = element_size / std::thread::hardware_concurrency();
  for (size_t i = 0; i < element_num; i++) {
    input_x_addr = inputs_addr + i * element_size;
    ParallelFor(element_size, per_unit_size, sequence_add_n);
  }

  return kAicpuKernelStateSucess;
}

uint32_t SequenceAddNKernel::DoCompute() {
  switch (input_data_type_) {
    case aicpuops::DataType::MS_INT32:
      return SequenceAddNTask<int>();
    case aicpuops::DataType::MS_INT64:
      return SequenceAddNTask<int64_t>();
    case aicpuops::DataType::MS_FLOAT32:
      return SequenceAddNTask<float>();
    case aicpuops::DataType::MS_FLOAT64:
      return SequenceAddNTask<double>();
    case aicpuops::DataType::MS_UINT32:
      return SequenceAddNTask<uint32_t>();
    case aicpuops::DataType::MS_UINT64:
      return SequenceAddNTask<uint64_t>();
    case aicpuops::DataType::MS_FLOAT16:
      return SequenceAddNTask<Eigen::half>();
    case aicpuops::DataType::MS_COMPLEX64:
      return SequenceAddNTask<std::complex<std::float_t>>();
    case aicpuops::DataType::MS_COMPLEX128:
      return SequenceAddNTask<std::complex<std::double_t>>();
    default:
      AICPU_LOGE("SequenceAddN kernel data type [%s] not support.", input_data_type_);
      return kAicpuKernelStateInvalid;
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t SequenceAddN(void *param) {
  aicpu::SequenceAddNKernel sequence_addn_kernel;
  return sequence_addn_kernel.Compute(param);
}
}
