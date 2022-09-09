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
  return ret;
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

  const int64_t x1_indices_num = inputs[kX1IndicesIdx]->size / (sizeof(S) * 2);
  const int64_t x2_indices_num = inputs[kX2IndicesIdx]->size / (sizeof(S) * 2);
  const int64_t out_indices_num = inputs[kOutIndicesIdx]->size / (sizeof(S) * 2);

  auto arrayHash = [fn = std::hash<int>{}](const std::array<int, 2> &arr) -> size_t {
    return std::accumulate(arr.begin(), arr.end(), 0u, [&](size_t acc, int num) { return (acc << 1) ^ fn(num); });
  };

  constexpr int dimension_difference = 2;
  std::unordered_map<std::array<int, 2>, int, decltype(arrayHash)> out_map(0, arrayHash);
  for (int i = 0; i < out_indices_num * dimension_difference; i += dimension_difference) {
    std::array<int, 2> index{};
    index[0] = out_indices[i];
    index[1] = out_indices[i + 1];
    out_map[index] = static_cast<int>(i / dimension_difference);
  }

  for (int i = 0; i < x1_indices_num * dimension_difference; i += dimension_difference) {
    std::array<int, 2> index{};
    index[0] = x1_indices[i];
    index[1] = x1_indices[i + 1];
    if (out_map.find(index) != out_map.end()) {
      dx1[static_cast<int>(i / dimension_difference)] = dout[out_map[index]];
    }
  }
  for (int i = 0; i < x2_indices_num * dimension_difference; i += dimension_difference) {
    std::array<int, 2> index{};
    index[0] = x2_indices[i];
    index[1] = x2_indices[i + 1];
    if (out_map.find(index) != out_map.end()) {
      dx2[static_cast<int>(i / dimension_difference)] = dout[out_map[index]];
    }
  }

  return true;
}

const std::vector<std::pair<KernelAttr, SparseAddGradCpuKernelMod::KernelRunFunc>>
  &SparseAddGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SparseAddGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseAddGradCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseAddGradCpuKernelMod::LaunchKernel<float, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseAddGrad, SparseAddGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
