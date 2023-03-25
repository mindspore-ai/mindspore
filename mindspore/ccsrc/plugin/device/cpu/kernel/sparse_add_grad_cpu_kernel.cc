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
#include <algorithm>
#include <utility>
#include <complex>
#include <set>
#include <map>
#include <functional>
#include <numeric>
#include <iterator>
#include <unordered_map>
#include "plugin/device/cpu/kernel/sparse_add_grad_cpu_kernel.h"
#include "mindspore/core/ops/grad/sparse_add_grad.h"

namespace mindspore {
namespace kernel {
// Value check constant
constexpr size_t kInputNum = 4;
constexpr size_t kOutputNum = 2;
// Input idx constant
constexpr size_t kDoutIdx = 0;
constexpr size_t kX1IndicesIdx = 1;
constexpr size_t kX2IndicesIdx = 2;
constexpr size_t kOutIndicesIdx = 3;
// Output idx constant
constexpr size_t kDx1Idx = 0;
constexpr size_t kDx2Idx = 1;

bool SparseAddGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseAddGrad>(base_operator);
  kernel_name_ = kernel_ptr->name();
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", input should be dout, x1_indices, x2_indices and out_indices total "
                  << kInputNum << " tensors, but get " << input_num;
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

void SparseAddGradCpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

int SparseAddGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret == KRET_UNKNOWN_OUT_SHAPE) {
    if (input_size_list_.size() != kInputNum) {
      MS_LOG(ERROR) << "Input size list should be " << kInputNum << ", but got " << input_size_list_.size();
      return KRET_RESIZE_FAILED;
    }
    auto dout_shape = inputs.at(kDoutIdx)->GetShapeVector();
    auto x1_indices_shape = inputs.at(kX1IndicesIdx)->GetShapeVector();
    auto x2_indices_shape = inputs.at(kX2IndicesIdx)->GetShapeVector();
    auto out_indices_shape = inputs.at(kOutIndicesIdx)->GetShapeVector();

    (void)std::transform(dout_shape.begin(), dout_shape.end(), std::back_inserter(dout_shape_), LongToSize);
    (void)std::transform(x1_indices_shape.begin(), x1_indices_shape.end(), std::back_inserter(x1_indices_shape_),
                         LongToSize);
    (void)std::transform(x2_indices_shape.begin(), x2_indices_shape.end(), std::back_inserter(x2_indices_shape_),
                         LongToSize);
    (void)std::transform(out_indices_shape.begin(), out_indices_shape.end(), std::back_inserter(out_indices_shape_),
                         LongToSize);

    auto dout_size_ = std::accumulate(dout_shape_.begin(), dout_shape_.end(), 1, std::multiplies<size_t>());
    auto x1_indices_size_ =
      std::accumulate(x1_indices_shape_.begin(), x1_indices_shape_.end(), 1, std::multiplies<size_t>());
    auto x2_indices_size_ =
      std::accumulate(x2_indices_shape_.begin(), x2_indices_shape_.end(), 1, std::multiplies<size_t>());
    auto out_indices_size_ =
      std::accumulate(out_indices_shape_.begin(), out_indices_shape_.end(), 1, std::multiplies<size_t>());

    input_size_list_.push_back(dout_size_);
    input_size_list_.push_back(x1_indices_size_);
    input_size_list_.push_back(x2_indices_size_);
    input_size_list_.push_back(out_indices_size_);
    output_size_list_.push_back(x1_indices_size_);
    output_size_list_.push_back(x2_indices_size_);
  }
  auto dims = inputs.at(0)->GetShapeVector()[1];
  if (dims >= 0) {
    indices_column_ = LongToSize(dims);
  }
  return ret;
}

template <typename T, typename S>
int SparseAddGradCpuKernelMod::CompareTwoIndices(const T &a_indices, const T &b_indices, const S *backprop_value,
                                                 int64_t *a_row, const int64_t b_row, const size_t dims, S *dx_value,
                                                 bool *idx_geq) {
  for (int64_t dim = 0; dim < SizeToLong(dims); dim++) {
    auto a_idx = a_indices[*a_row * dims + dim];
    auto b_idx = b_indices[b_row * dims + dim];
    if (a_idx < b_idx) {
      *idx_geq = false;
      *a_row += 1;
      return -1;
    } else if (a_idx > b_idx) {
      return 1;
    }
  }
  dx_value[*a_row] = backprop_value[b_row];
  *a_row += 1;
  return 0;
}

template <typename T, typename S>
bool SparseAddGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the number of inputs should be " << kInputNum << ", but got "
                      << inputs.size() << " input(s).";
  }
  if (outputs.size() != kOutputNum) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the number of inputs should be " << kOutputNum << ", but got "
                      << outputs.size() << " output(s).";
  }
  // Inputs
  const auto dout = reinterpret_cast<T *>(inputs[kDoutIdx]->addr);
  const auto x1_indices = reinterpret_cast<S *>(inputs[kX1IndicesIdx]->addr);
  const auto x2_indices = reinterpret_cast<S *>(inputs[kX2IndicesIdx]->addr);
  const auto out_indices = reinterpret_cast<S *>(inputs[kOutIndicesIdx]->addr);
  // Outputs
  auto dx1 = reinterpret_cast<T *>(outputs[kDx1Idx]->addr);
  auto dx2 = reinterpret_cast<T *>(outputs[kDx2Idx]->addr);

  const int64_t x1_indices_num = inputs[kX1IndicesIdx]->size / (sizeof(S) * indices_column_);
  const int64_t x2_indices_num = inputs[kX2IndicesIdx]->size / (sizeof(S) * indices_column_);
  const int64_t out_indices_num = inputs[kOutIndicesIdx]->size / (sizeof(S) * indices_column_);

  memset_s(dx1, sizeof(T) * x1_indices_num, 0, sizeof(T) * x1_indices_num);
  memset_s(dx2, sizeof(T) * x2_indices_num, 0, sizeof(T) * x2_indices_num);

  int64_t i = 0;
  int64_t j = 0;
  int64_t k = 0;
  bool a_idx_geq;
  bool b_idx_geq;

  while (i < x1_indices_num && j < x2_indices_num && k < out_indices_num) {
    a_idx_geq = b_idx_geq = true;
    CompareTwoIndices(x1_indices, out_indices, dout, &i, k, indices_column_, dx1, &a_idx_geq);
    CompareTwoIndices(x2_indices, out_indices, dout, &j, k, indices_column_, dx2, &b_idx_geq);
    if (a_idx_geq && b_idx_geq) {
      k += 1;
    }
  }
  while (i < x1_indices_num && k < out_indices_num) {
    a_idx_geq = true;
    CompareTwoIndices(x1_indices, out_indices, dout, &i, k, indices_column_, dx1, &a_idx_geq);
    if (a_idx_geq) {
      k += 1;
    }
  }

  while (j < x2_indices_num && k < out_indices_num) {
    b_idx_geq = true;
    CompareTwoIndices(x2_indices, out_indices, dout, &j, k, indices_column_, dx2, &b_idx_geq);
    if (b_idx_geq) {
      k += 1;
    }
  }

  return true;
}

#define CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(ms_index_type, ms_value_type, index_type, value_type) \
  {                                                                                               \
    KernelAttr()                                                                                  \
      .AddInputAttr(ms_value_type)                                                                \
      .AddInputAttr(ms_index_type)                                                                \
      .AddInputAttr(ms_index_type)                                                                \
      .AddInputAttr(ms_index_type)                                                                \
      .AddOutputAttr(ms_value_type)                                                               \
      .AddOutputAttr(ms_value_type),                                                              \
      &SparseAddGradCpuKernelMod::LaunchKernel<value_type, index_type>                            \
  }

const std::vector<std::pair<KernelAttr, SparseAddGradCpuKernelMod::KernelRunFunc>>
  &SparseAddGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SparseAddGradCpuKernelMod::KernelRunFunc>> func_list = {
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt8, int64_t, int8_t),
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt16, int64_t, int16_t),
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t),
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeComplex64, int64_t, std::complex<float>),
    CPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeComplex128, int64_t, std::complex<double>),
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseAddGrad, SparseAddGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
