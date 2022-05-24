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
#include "plugin/device/cpu/kernel/sparse_add_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <set>
#include <map>
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/sparse_add.h"

namespace mindspore {
namespace kernel {
// Value check constant
constexpr size_t kInputNum = 4;
constexpr size_t kOutputNum = 2;
constexpr size_t kNumOfColumn = 2;
// Input idx constant
constexpr size_t kAIndicesIdx = 0;
constexpr size_t kAValuesIdx = 1;
constexpr size_t kBIndicesIdx = 2;
constexpr size_t kBValuesIdx = 3;
// Output idx constant
constexpr size_t kSumIndicesIdx = 0;
constexpr size_t kSumValuesIdx = 1;

bool SparseAddCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  outputs_ = outputs;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseAdd>(base_operator);
  thresh_ = kernel_ptr->get_thresh();
  kernel_name_ = kernel_ptr->name();
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", input should be a_indices, a_values, b_indices and b_values total "
                  << kInputNum << " tensors, but get " << input_num;
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  auto dense_shape = kernel_ptr->get_a_dense_shape();
  row_ = LongToSize(dense_shape[0]);
  dense_size_ = row_ * LongToSize(dense_shape[1]) * GetTypeByte(TypeIdToType(types_[1]));
  is_need_retrieve_output_shape_ = true;
  for (size_t i = 0; i < kOutputNum; i++) {
    auto dtype = inputs[i]->GetDtype();
    (void)types_.emplace_back(dtype);
  }
  return true;
}

int SparseAddCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret == KRET_UNKNOWN_OUT_SHAPE) {
    if (input_size_list_.size() != kInputNum) {
      MS_LOG(ERROR) << "Input size list should be " << kInputNum << ", but got " << input_size_list_.size();
      return KRET_RESIZE_FAILED;
    }
    auto max_indices_out_size =
      std::min(input_size_list_[kAIndicesIdx] + input_size_list_[kBIndicesIdx], dense_size_ * 2);
    auto max_value_out_size = std::min(input_size_list_[kAValuesIdx] + input_size_list_[kBValuesIdx], dense_size_);
    output_size_list_.emplace_back(max_indices_out_size);
    output_size_list_.emplace_back(max_value_out_size);
  }
  return ret;
}

template <typename T>
int SparseAddCpuKernelMod::CompareTowIndices(const T &a_indices, const T &b_indices, int64_t a_row, int64_t b_row,
                                             const size_t dims) {
  for (size_t dim = 0; dim < dims; dim++) {
    auto a_idx = a_indices[a_row * 2 + dim];
    auto b_idx = b_indices[b_row * 2 + dim];
    if (a_idx < b_idx) {
      return -1;
    } else if (a_idx > b_idx) {
      return 1;
    }
  }
  return 0;
}

template <typename T, typename S>
bool SparseAddCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << kInputNum << ", but got "
                      << inputs.size() << " input(s).";
  }
  if (outputs.size() != kOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be " << kOutputNum << ", but got "
                      << outputs.size() << " output(s).";
  }
  // Inputs
  const auto a_indices = reinterpret_cast<T *>(inputs[kAIndicesIdx]->addr);
  const auto a_values = reinterpret_cast<S *>(inputs[kAValuesIdx]->addr);
  const auto b_indices = reinterpret_cast<T *>(inputs[kBIndicesIdx]->addr);
  const auto b_values = reinterpret_cast<S *>(inputs[kBValuesIdx]->addr);
  // Outputs
  auto sum_indices = reinterpret_cast<T *>(outputs[kSumIndicesIdx]->addr);
  auto sum_values = reinterpret_cast<S *>(outputs[kSumValuesIdx]->addr);

  const int64_t a_indices_num = inputs[kAIndicesIdx]->size / ((sizeof(T)) * 2);
  const int64_t b_indices_num = inputs[kBIndicesIdx]->size / ((sizeof(T)) * 2);

  // Use double pointer to calculate the sum of two inputs
  int64_t i = 0, j = 0;
  S sum_ab = 0;
  std::vector<std::pair<bool, int64_t>> whole_indices;
  std::vector<S> whole_values;
  whole_indices.reserve(a_indices_num + b_indices_num);
  while (i < a_indices_num && j < b_indices_num) {
    switch (CompareTowIndices(a_indices, b_indices, i, j, kNumOfColumn)) {
      case -1:
        whole_indices.emplace_back(true, i);
        whole_values.push_back(a_values[i]);
        i += 1;
        break;
      case 0:
        sum_ab = a_values[i] + b_values[j];
        if (thresh_ <= std::abs(sum_ab)) {
          whole_indices.emplace_back(true, i);
          whole_values.push_back(sum_ab);
        }
        i += 1;
        j += 1;
        break;
      case 1:
        whole_indices.emplace_back(false, j);
        whole_values.push_back(b_values[j]);
        j += 1;
        break;
    }
  }

  if (i < a_indices_num) {
    while (i < a_indices_num) {
      whole_indices.emplace_back(true, i);
      whole_values.push_back(a_values[i]);
      i += 1;
    }
  } else {
    while (j < b_indices_num) {
      whole_indices.emplace_back(false, j);
      whole_values.push_back(b_values[j]);
      j += 1;
    }
  }

  for (size_t num = 0; num < whole_indices.size(); num++) {
    auto copy_from_a = whole_indices[num].first;
    auto index_from_input = whole_indices[num].second;
    if (copy_from_a) {
      for (size_t column = 0; column < kNumOfColumn; column++) {
        sum_indices[num * kNumOfColumn + column] = a_indices[index_from_input * kNumOfColumn + column];
      }
    } else {
      for (size_t column = 0; column < kNumOfColumn; column++) {
        sum_indices[num * kNumOfColumn + column] = b_indices[index_from_input * kNumOfColumn + column];
      }
    }
    sum_values[num] = whole_values[num];
  }

  // Update output shape and type
  std::vector<int64_t> out_indices_shape;
  std::vector<int64_t> out_values_shape;
  (void)out_indices_shape.emplace_back(SizeToLong(whole_indices.size()));
  (void)out_indices_shape.emplace_back(SizeToLong(kNumOfColumn));
  (void)out_values_shape.emplace_back(SizeToLong(whole_values.size()));
  outputs_[kSumIndicesIdx]->SetShapeVector(out_indices_shape);
  outputs_[kSumValuesIdx]->SetShapeVector(out_values_shape);

  return true;
}

#define CPU_SPARSE_ADD_KERNEL_REGISTER(ms_index_type, ms_value_type, index_type, value_type) \
  {                                                                                          \
    KernelAttr()                                                                             \
      .AddInputAttr(ms_index_type)                                                           \
      .AddInputAttr(ms_value_type)                                                           \
      .AddInputAttr(ms_index_type)                                                           \
      .AddInputAttr(ms_value_type)                                                           \
      .AddOutputAttr(ms_index_type)                                                          \
      .AddOutputAttr(ms_value_type),                                                         \
      &SparseAddCpuKernelMod::LaunchKernel<index_type, value_type>                           \
  }

const std::vector<std::pair<KernelAttr, SparseAddCpuKernelMod::KernelRunFunc>> &SparseAddCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, SparseAddCpuKernelMod::KernelRunFunc>> func_list = {
    // float values
    CPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat32, int, float),
    // double values
    CPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat64, int, double),
    // int values
    CPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int, int),
    // int64 values
    CPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
    // int16 values
    CPU_SPARSE_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt16, int, int16_t),
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseAdd, SparseAddCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
