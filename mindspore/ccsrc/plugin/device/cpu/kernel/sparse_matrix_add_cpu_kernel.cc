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
#include "plugin/device/cpu/kernel/sparse_matrix_add_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <set>
#include <map>
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/sparse_matrix_add.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMatrixDims = 2;
constexpr size_t kInputNum = 8;
constexpr size_t kOutputNum = 3;
constexpr size_t kAIndptrIdx = 0;
constexpr size_t kAIndicesIdx = 1;
constexpr size_t kAValuesIdx = 2;
constexpr size_t kBIndptrIdx = 3;
constexpr size_t kBIndicesIdx = 4;
constexpr size_t kBValuesIdx = 5;
constexpr size_t kAlphaIdx = 6;
constexpr size_t kBetaIdx = 7;
constexpr size_t kOutIndptr = 0;
constexpr size_t kOutIndices = 1;
constexpr size_t kOutValue = 2;
bool SparseMatirxAddCpuKernelMod::Init(const BaseOperatorPtr &base_operater, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  outputs_ = outputs;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseMatrixAdd>(base_operater);
  kernel_name_ = kernel_ptr->name();
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For " << kernel_name_
                  << ", input should be A(indptr, indices, values), B(indptr, indeices, values), alpha, beta total "
                  << kInputNum << " tensors, but get " << input_num;
    return false;
  }
  auto dense_shape = kernel_ptr->get_dense_shape();
  if (dense_shape.size() != kMatrixDims) {
    MS_LOG(ERROR) << "The supported dims for input is " << kMatrixDims << "-D, but get " << dense_shape.size();
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(WARNING) << kernel_name_ << " does not support this data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  row_ = LongToSize(dense_shape[0]);
  for (size_t i = 0; i < kOutputNum; i++) {
    auto dtype = inputs[i]->GetDtype();
    (void)types_.emplace_back(dtype);
  }
  return true;
}

int SparseMatirxAddCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  return NativeCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
}

template <typename T, typename S>
bool SparseMatirxAddCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << kInputNum << ", but got "
                      << inputs.size() << " input(s).";
  }
  if (outputs.size() != kOutputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be " << kOutputNum << ", but got "
                      << outputs.size() << " output(s).";
  }

  const auto a_indptr = reinterpret_cast<T *>(inputs[kAIndptrIdx]->addr);
  const auto a_indices = reinterpret_cast<T *>(inputs[kAIndicesIdx]->addr);
  const auto a_values = reinterpret_cast<S *>(inputs[kAValuesIdx]->addr);
  const auto b_indptr = reinterpret_cast<T *>(inputs[kBIndptrIdx]->addr);
  const auto b_indices = reinterpret_cast<T *>(inputs[kBIndicesIdx]->addr);
  const auto b_values = reinterpret_cast<S *>(inputs[kBValuesIdx]->addr);
  const auto alpha = reinterpret_cast<S *>(inputs[kAlphaIdx]->addr);
  const auto beta = reinterpret_cast<S *>(inputs[kBetaIdx]->addr);
  auto c_indptr = reinterpret_cast<T *>(outputs[kAIndptrIdx]->addr);
  auto c_indices = reinterpret_cast<T *>(outputs[kAIndicesIdx]->addr);
  auto c_values = reinterpret_cast<S *>(outputs[kAValuesIdx]->addr);

  // Do the compute: C = alpha * A + beta * B.
  c_indptr[0] = 0;
  std::set<T> index_set;
  size_t c_idx = 0;
  T a_v = 0;
  T b_v = 0;
  size_t a_v_idx = 0;
  size_t b_v_idx = 0;
  auto task = [this, &a_indptr, &a_indices, &a_values, &b_indptr, &b_indices, &b_values, &alpha, &beta, &c_indptr,
               &c_indices, &c_values, &index_set, &c_idx, &a_v, &b_v, &a_v_idx, &b_v_idx](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t max_mask_len = a_indptr[i + 1] - a_indptr[i] + b_indptr[i + 1] - b_indptr[i];
      // Masks for recording the valid location.
      std::vector<bool> mask_a(max_mask_len, false);
      std::vector<bool> mask_b(max_mask_len, false);
      for (size_t j = static_cast<size_t>(a_indptr[i]); j < static_cast<size_t>(a_indptr[i + 1]); j++) {
        (void)index_set.insert(a_indices[j]);
        mask_a[a_indices[j]] = true;
      }
      for (size_t k = static_cast<size_t>(b_indptr[i]); k < static_cast<size_t>(b_indptr[i + 1]); k++) {
        (void)index_set.insert(b_indices[k]);
        mask_b[b_indices[k]] = true;
      }
      // index_set.size() are the valid numbers to set indptr.
      c_indptr[i + 1] = c_indptr[i] + static_cast<T>(index_set.size());
      for (auto it = index_set.begin(); it != index_set.end(); it++) {
        if (mask_a[*it]) {
          // Independent cursor for indeices to get value. Increase the cursor once used.
          a_v = a_values[a_v_idx];
          a_v_idx++;
        }
        if (mask_b[*it]) {
          b_v = b_values[b_v_idx];
          b_v_idx++;
        }
        c_values[c_idx] = alpha[0] * a_v + beta[0] * b_v;
        c_indices[c_idx] = *it;
        c_idx++;
        b_v = 0;  // Reset the tmp value, real number or zero.
        a_v = 0;
      }
      index_set.clear();
    }
  };
  ParallelLaunchAutoSearch(task, row_, this, &parallel_search_info_);
  // Update output shape and type
  std::vector<int64_t> out_indptr_shape;
  std::vector<int64_t> out_indices_shape;
  std::vector<int64_t> out_values_shape;
  (void)out_indptr_shape.emplace_back(SizeToLong(row_ + 1));
  (void)out_indices_shape.emplace_back(SizeToLong(c_idx));
  (void)out_values_shape.emplace_back(SizeToLong(c_idx));
  outputs_[kOutIndptr]->SetShapeVector(out_indptr_shape);
  outputs_[kOutIndices]->SetShapeVector(out_indices_shape);
  outputs_[kOutValue]->SetShapeVector(out_values_shape);
  return true;
}

#define CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(ms_index_type, ms_value_type, index_type, value_type) \
  {                                                                                                 \
    KernelAttr()                                                                                    \
      .AddInputAttr(ms_index_type)                                                                  \
      .AddInputAttr(ms_index_type)                                                                  \
      .AddInputAttr(ms_value_type)                                                                  \
      .AddInputAttr(ms_index_type)                                                                  \
      .AddInputAttr(ms_index_type)                                                                  \
      .AddInputAttr(ms_value_type)                                                                  \
      .AddInputAttr(ms_value_type)                                                                  \
      .AddInputAttr(ms_value_type)                                                                  \
      .AddOutputAttr(ms_index_type)                                                                 \
      .AddOutputAttr(ms_index_type)                                                                 \
      .AddOutputAttr(ms_value_type),                                                                \
      &SparseMatirxAddCpuKernelMod::LaunchKernel<index_type, value_type>                            \
  }

std::vector<std::pair<KernelAttr, SparseMatirxAddCpuKernelMod::SparseMatrixAddFunc>>
  SparseMatirxAddCpuKernelMod::func_list_ = {
    // float values
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeFloat32, int16_t, float),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat32, int, float),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
    // double values
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeFloat64, int16_t, double),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat64, int, double),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
    // int values
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt32, int16_t, int),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int, int),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t, int),
    // int64 values
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    // int16 values
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt16, int16_t, int16_t),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt16, int, int16_t),
    CPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt16, int64_t, int16_t)};

std::vector<KernelAttr> SparseMatirxAddCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseMatrixAddFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseMatrixAdd, SparseMatirxAddCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
