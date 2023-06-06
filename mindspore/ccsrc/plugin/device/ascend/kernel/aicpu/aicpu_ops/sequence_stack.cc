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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/sequence_stack.h"
#include <securec.h>
#include <string>
#include <thread>
#include <complex>
#include <algorithm>
#include "proto/aicpu_tensor.pb.h"
#include "common/atomic_op.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "Eigen/Core"
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
constexpr size_t kSequenceStackInputNum = 1;
constexpr size_t kSequenceStackOutputNum = 1;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;

uint32_t SequenceStackKernel::ParseKernelParam() {
  if (node_def_.inputs_size() != kSequenceStackInputNum) {
    AICPU_LOGE("For 'SequenceStack', input number must be 1, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }

  if (node_def_.outputs_size() != kSequenceStackOutputNum) {
    AICPU_LOGE("For 'SequenceStack', output number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }
  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  input_data_type_ = static_cast<aicpuops::DataType>(input_tensor.tensor_type());
  tuple_shape = GetShape(input_tensor.tensor_shape());
  output_data_size_ = GetTensorMemSizeByShape(node_def_.outputs(kDim0));

  std::vector<int64_t> shape_vec_item;
  std::copy(tuple_shape.begin() + 1, tuple_shape.end(), std::back_inserter(shape_vec_item));

  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  axis_ = attrs["axis"].i();
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(shape_vec_item.size()) + 1;
  }
  dims_behind_axis_ = 1;
  // calculate elements while dim >= axis
  for (size_t i = IntToSize(axis_); i < shape_vec_item.size(); i++) {
    dims_behind_axis_ *= static_cast<int64_t>(shape_vec_item[i]);
  }
  return kAicpuKernelStateSucess;
}

template <typename T>
uint32_t SequenceStackKernel::SequenceStackTask() {
  const auto inputs_addr = reinterpret_cast<T *>(io_addrs_[kDim0]);
  auto *output_addr = reinterpret_cast<T *>(io_addrs_[kDim1]);
  auto element_num = LongToSize(tuple_shape[0]);
  auto element_size = output_data_size_ / sizeof(T);
  auto cp_ret = memset_s(output_addr, output_data_size_, 0x0, output_data_size_);
  if (cp_ret != EOK) {
    AICPU_LOGE("For 'SequenceStack',  memset for output error, errorno: %d, size: %d.", cp_ret, output_data_size_);
    return kAicpuKernelStateInvalid;
  }
  size_t element_index_size =
    std::accumulate(tuple_shape.begin() + 1, tuple_shape.end(), 1, std::multiplies<int64_t>());
  const int64_t per_unit_size = element_size / dims_behind_axis_;
  auto copy_num = dims_behind_axis_;
  auto copy_size = copy_num * sizeof(T);
  auto tasks = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; ++pos) {
      if (element_num == 0) {
        AICPU_LOGE("For 'SequenceStack',  the element of inputs must be greater than 0, but got: %d.", element_num);
      }
      size_t cur_input_index = pos % element_num;
      size_t local_idx = pos / element_num;
      (void)memcpy_s(output_addr + dims_behind_axis_ * pos, copy_size,
                     inputs_addr + cur_input_index * element_index_size + dims_behind_axis_ * local_idx, copy_size);
    }
  };
  ParallelFor(element_size, per_unit_size, tasks);

  return kAicpuKernelStateSucess;
}

uint32_t SequenceStackKernel::DoCompute() {
  switch (input_data_type_) {
    case aicpuops::DataType::MS_INT32:
      return SequenceStackTask<int>();
    case aicpuops::DataType::MS_INT64:
      return SequenceStackTask<int64_t>();
    case aicpuops::DataType::MS_FLOAT32:
      return SequenceStackTask<float>();
    case aicpuops::DataType::MS_FLOAT64:
      return SequenceStackTask<double>();
    case aicpuops::DataType::MS_UINT32:
      return SequenceStackTask<uint32_t>();
    case aicpuops::DataType::MS_UINT64:
      return SequenceStackTask<uint64_t>();
    case aicpuops::DataType::MS_FLOAT16:
      return SequenceStackTask<Eigen::half>();
    case aicpuops::DataType::MS_COMPLEX64:
      return SequenceStackTask<std::complex<std::float_t>>();
    case aicpuops::DataType::MS_COMPLEX128:
      return SequenceStackTask<std::complex<std::double_t>>();
    case aicpuops::DataType::MS_BOOL:
      return SequenceStackTask<bool>();
    default:
      AICPU_LOGE("SequenceStack kernel data type [%s] not support.", input_data_type_);
      return kAicpuKernelStateInvalid;
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t SequenceStack(void *param) {
  aicpu::SequenceStackKernel sequence_stack_kernel;
  return sequence_stack_kernel.Compute(param);
}
}
