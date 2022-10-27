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

#include "mindspore/ccsrc/plugin/device/cpu/kernel/non_zero_cpu_kernel.h"
#include <algorithm>
#include <typeinfo>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
constexpr size_t kInputMinDim = 1;
constexpr size_t kOutputDim = 2;
}  // namespace

bool NonZeroCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  is_need_retrieve_output_shape_ = true;  // NonZero is a dynamic shape operator.
  data_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  index_size_ = abstract::TypeIdSize(outputs[kIndex0]->GetDtype());
  return true;
}

int NonZeroCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  outputs_ = outputs;
  auto input_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>{});
  if (input_size_ == 0) {
    return KRET_UNKNOWN_SHAPE;
  }
  input_rank_ = input_shape_.size();

  input_size_list_.push_back(input_size_ * data_size_);
  output_size_list_.push_back(input_size_ * input_shape_.size() * index_size_);
  return KRET_OK;
}

void NonZeroCpuKernelMod::ResetResource() noexcept {
  real_output_size_ = 0;
  input_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool NonZeroCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (input_size_ == 0) {
    return true;
  }

  auto input_addr = static_cast<T *>(inputs[0]->addr);
  auto output_addr = static_cast<int64_t *>(outputs[0]->addr);
  real_output_size_ = NonZeroCompute(input_addr, output_addr, input_size_);
  return true;
}

void NonZeroCpuKernelMod::SyncData() {
  std::vector<int64_t> new_output_shape = {SizeToLong(real_output_size_), SizeToLong(input_shape_.size())};
  outputs_[kIndex0]->SetShapeVector(new_output_shape);
}

template <typename T>
size_t NonZeroCpuKernelMod::NonZeroCompute(const T *input, int64_t *output, size_t input_num) {
  size_t non_zero_count = 0;
  std::vector<size_t> dim_strides(input_rank_, 1);

  for (size_t i = input_rank_ - 1; i >= 1; --i) {
    dim_strides[i - 1] = dim_strides[i] * input_shape_[i];
  }

  for (size_t elem_i = 0; elem_i < input_num; ++elem_i) {
    auto zero = static_cast<T>(0);
    if constexpr (std::is_same_v<T, double>) {
      if (common::IsDoubleEqual(input[elem_i], zero)) {
        continue;
      }
    } else {
      if constexpr (std::is_same_v<T, float>) {
        if (common::IsFloatEqual(input[elem_i], zero)) {
          continue;
        }
      } else {
        if (input[elem_i] == zero) {
          continue;
        }
      }
    }
    size_t index = elem_i;
    for (size_t pos_j = 0; pos_j < input_rank_; ++pos_j) {
      output[non_zero_count * input_rank_ + pos_j] = static_cast<int64_t>(index / dim_strides[pos_j]);
      index %= dim_strides[pos_j];
    }
    non_zero_count++;
  }
  return non_zero_count;
}

std::vector<std::pair<KernelAttr, NonZeroCpuKernelMod::NonZeroFunc>> NonZeroCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> NonZeroCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NonZeroFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NonZero, NonZeroCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
