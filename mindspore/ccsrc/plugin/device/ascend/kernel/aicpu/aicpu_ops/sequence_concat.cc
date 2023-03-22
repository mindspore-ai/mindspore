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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/sequence_concat.h"
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

ShapeVector FlatShapeByAxis(const ShapeVector &shape, int axis) {
  if (axis < 0) {
    axis = axis + SizeToInt(shape.size());
  }
  int64_t dim_row = 1;
  int64_t dim_col = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (SizeToInt(i) < axis) {
      dim_row *= shape[i];
    } else {
      dim_col *= shape[i];
    }
  }

  return ShapeVector{dim_row, dim_col};
}
}  // namespace
constexpr size_t kSequenceConcatInputNum = 1;
constexpr size_t kSequenceConcatOutputNum = 1;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;

uint32_t SequenceConcatKernel::ParseKernelParam() {
  if (node_def_.inputs_size() != kSequenceConcatInputNum) {
    AICPU_LOGE("For 'SequenceConcat', input number must be 1, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }

  if (node_def_.outputs_size() != kSequenceConcatOutputNum) {
    AICPU_LOGE("For 'SequenceConcat', output number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }
  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  input_data_type_ = static_cast<aicpuops::DataType>(input_tensor.tensor_type());
  tuple_shape = GetShape(input_tensor.tensor_shape());
  input_data_size_ = GetTensorMemSizeByShape(node_def_.inputs(kDim0));
  output_data_size_ = GetTensorMemSizeByShape(node_def_.outputs(kDim0));

  std::vector<int64_t> shape_vec_item;
  std::copy(tuple_shape.begin() + 1, tuple_shape.end(), std::back_inserter(shape_vec_item));
  input_shapes_.clear();
  for (int64_t i = 0; i < tuple_shape[0]; ++i) {
    input_shapes_.push_back(shape_vec_item);
  }
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  axis_ = attrs["axis"].i();
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(input_shapes_[0].size());
  }

  input_flat_shape_list_.clear();
  for (int64_t i = 0; i < tuple_shape[0]; i++) {
    auto input_shape_i = input_shapes_[i];
    auto flat_shape = FlatShapeByAxis(input_shape_i, axis_);
    (void)input_flat_shape_list_.emplace_back(flat_shape);
  }

  output_dim_ = 0;
  offset_.clear();
  for (int64_t j = 0; j < tuple_shape[0]; ++j) {
    offset_.push_back(output_dim_);
    output_dim_ += LongToSize(input_flat_shape_list_[j][1]);
  }

  return kAicpuKernelStateSucess;
}

template <typename T>
uint32_t SequenceConcatKernel::SequenceConcatTask() {
  const auto inputs_addr = reinterpret_cast<T *>(io_addrs_[kDim0]);
  auto *output_addr = reinterpret_cast<T *>(io_addrs_[kDim1]);
  auto element_num = LongToSize(tuple_shape[0]);
  auto element_size = output_data_size_ / sizeof(T);
  auto cp_ret = memset_s(output_addr, output_data_size_, 0x0, output_data_size_);
  if (cp_ret != EOK) {
    AICPU_LOGE("For 'SequenceConcat',  memset for output error, errorno: %d, size: %d.", cp_ret, output_data_size_);
    return kAicpuKernelStateInvalid;
  }

  size_t element_index_size =
    std::accumulate(tuple_shape.begin() + 1, tuple_shape.end(), 1, std::multiplies<int64_t>());
  std::vector<T *> input_addr_list;
  for (uint64_t j = 0; j < element_num; ++j) {
    auto *tmp_addr = reinterpret_cast<T *>(inputs_addr + j * element_index_size);
    (void)input_addr_list.emplace_back(tmp_addr);
  }
  if (input_flat_shape_list_.empty() || input_flat_shape_list_[0].empty()) {
    return true;
  }
  const int64_t per_unit_size = element_size / std::thread::hardware_concurrency();
  auto tasks = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; ++pos) {
      if (element_num == 0) {
        AICPU_LOGE("For 'SequenceConcat',  the element of inputs must be greater than 0, but got: %d.", element_num);
      }
      size_t i = pos / element_num;
      size_t j = pos % element_num;

      if (input_flat_shape_list_[j][1] == 0) {
        continue;
      }
      auto copy_num = LongToSize(input_flat_shape_list_[j][1]);
      auto copy_size = copy_num * sizeof(T);
      auto offset = copy_num * i;
      auto output_ptr = output_addr + i * output_dim_ + offset_[j];
      auto ret = memcpy_s(output_ptr, copy_size, input_addr_list[j] + offset, copy_size);
      if (ret != EOK) {
        AICPU_LOGE("For 'SequenceConcat', memcpy_s failed. Error no: %d", ret);
      }
    }
  };
  ParallelFor(element_size, per_unit_size, tasks);

  return kAicpuKernelStateSucess;
}

uint32_t SequenceConcatKernel::DoCompute() {
  switch (input_data_type_) {
    case aicpuops::DataType::MS_INT32:
      return SequenceConcatTask<int>();
    case aicpuops::DataType::MS_INT64:
      return SequenceConcatTask<int64_t>();
    case aicpuops::DataType::MS_FLOAT32:
      return SequenceConcatTask<float>();
    case aicpuops::DataType::MS_FLOAT64:
      return SequenceConcatTask<double>();
    case aicpuops::DataType::MS_UINT32:
      return SequenceConcatTask<uint32_t>();
    case aicpuops::DataType::MS_UINT64:
      return SequenceConcatTask<uint64_t>();
    case aicpuops::DataType::MS_FLOAT16:
      return SequenceConcatTask<Eigen::half>();
    case aicpuops::DataType::MS_COMPLEX64:
      return SequenceConcatTask<std::complex<std::float_t>>();
    case aicpuops::DataType::MS_COMPLEX128:
      return SequenceConcatTask<std::complex<std::double_t>>();
    case aicpuops::DataType::MS_BOOL:
      return SequenceConcatTask<bool>();
    default:
      AICPU_LOGE("SequenceConcat kernel data type [%s] not support.", input_data_type_);
      return kAicpuKernelStateInvalid;
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t SequenceConcat(void *param) {
  aicpu::SequenceConcatKernel sequence_concat_kernel;
  return sequence_concat_kernel.Compute(param);
}
}
