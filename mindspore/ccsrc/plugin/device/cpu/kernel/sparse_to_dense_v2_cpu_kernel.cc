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

#include "plugin/device/cpu/kernel/sparse_to_dense_v2_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseToDenseV2TwoDims = 2;
constexpr size_t kSparseToDenseV2OneDim = 1;
constexpr size_t kSparseToDenseV2ZeroDim = 0;
constexpr size_t kSize0 = 0;
constexpr size_t kSize1 = 1;
constexpr size_t kSize2 = 2;
constexpr size_t kSize3 = 3;
constexpr size_t kSize4 = 4;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
#define ADD_KERNEL(t1, t2, t3, t4, t5) \
  KernelAttr()                         \
    .AddInputAttr(kNumberType##t1)     \
    .AddInputAttr(kNumberType##t2)     \
    .AddInputAttr(kNumberType##t3)     \
    .AddInputAttr(kNumberType##t4)     \
    .AddOutputAttr(kNumberType##t5)
}  // namespace
bool SparseToDenseV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}
int SparseToDenseV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto indices_shape = inputs.at(kIndex0)->GetShapeVector();
  indices_shape_ = Convert2SizeT(indices_shape);
  indices_dims_ = indices_shape_.size();
  auto output_shape = inputs.at(kIndex1)->GetShapeVector();
  output_shape_ = Convert2SizeT(output_shape);
  auto values_shape = inputs.at(kIndex2)->GetShapeVector();
  if (values_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For the third input parameter, the size of it should not be empty.";
  }
  values_size_ = LongToSize(values_shape[0]);
  if (indices_shape_.size() == 0) {
    if (values_shape.size() != 0 && values_shape[0] != 1) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the indices_shape[0] is 1"
                               << " should match the the values element " << values_size_ << ".";
    }
  } else {
    if (values_shape.size() != 0) {
      if (indices_shape_[0] != values_size_) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the indices_shape[0] " << indices_shape_[0]
                                 << " should match the the values element " << values_size_ << ".";
      }
    }
  }
  return KRET_OK;
}
template <typename I, typename T>
void SparseToDenseV2CpuKernelMod::CheckValidate(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs, const bool dim_flag) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSize4, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSize1, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
  }
  auto ret = memset_s(outputs[0]->addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret;
  }
  const auto *indices_addr = static_cast<I *>(inputs[kIndex0]->addr);
  const auto *output_shape_addr = static_cast<I *>(inputs[kIndex1]->addr);
  bool valid = true;
  bool different = false;
  bool increasing = true;
  size_t indices_shape_dim0 = static_cast<size_t>(indices_shape_[0]);
  size_t indices_shape_dim1 = static_cast<size_t>(1);
  if (dim_flag) {
    indices_shape_dim1 = static_cast<size_t>(indices_shape_[1]);
  }
  for (size_t k = 0; k < indices_shape_dim1; ++k) {
    size_t index = k;
    if (indices_addr[index] < 0 || indices_addr[index] >= output_shape_addr[index]) {
      valid = false;
    }
  }
  if (!valid) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the indices is out of bounds.";
  }
  for (size_t i = 1; i < indices_shape_dim0; ++i) {
    for (size_t j = 0; j < indices_shape_dim1; ++j) {
      size_t index1 = i * indices_shape_dim1 + j;
      size_t index2 = (i - 1) * indices_shape_dim1 + j;
      if (indices_addr[index1] < 0 || indices_addr[index1] >= output_shape_addr[j]) {
        valid = false;
      }
      I diff = indices_addr[index1] - indices_addr[index2];
      if (diff > 0) {
        different = true;
      }
      if (!different && diff < 0) {
        increasing = false;
      }
    }
    if (!valid) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the indices is out of bounds.";
    }
    if (!increasing) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the indices is out of order.";
    }
    if (!different) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the indices is repeated";
    }
  }
}
template <typename I, typename T>
bool SparseToDenseV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &workspace,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  if (validate_indices_ == true && indices_dims_ == kSparseToDenseV2TwoDims) {
    (void)SparseToDenseV2CpuKernelMod::CheckValidate<I, T>(inputs, outputs, true);
  } else if (validate_indices_ == true && indices_dims_ == kSparseToDenseV2OneDim) {
    (void)SparseToDenseV2CpuKernelMod::CheckValidate<I, T>(inputs, outputs, false);
  }
  const auto *indices_addr = static_cast<I *>(inputs[kIndex0]->addr);
  const auto *output_shape_addr = static_cast<I *>(inputs[kIndex1]->addr);
  const auto *values_addr = static_cast<T *>(inputs[kIndex2]->addr);
  const auto *default_value_addr = static_cast<T *>(inputs[kIndex3]->addr);
  auto *output_addr = static_cast<T *>(outputs[0]->addr);
  const size_t indices_length = inputs[kIndex0]->size / sizeof(I);
  const size_t output_length = outputs[0]->size / sizeof(T);
  const size_t values_length = inputs[kIndex2]->size / sizeof(T);
  size_t rank = output_shape_[0];
  for (size_t p = 0; p < output_length; ++p) {
    output_addr[p] = default_value_addr[0];
  }
  if (indices_dims_ == kSparseToDenseV2ZeroDim) {
    size_t out_index = 0;
    int index = indices_addr[0];
    if (index >= output_shape_addr[0] || index < 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the  0th value in "
                               << "0th dimension index: " << index << " of 'output' out of bounds: [0, "
                               << output_shape_addr[0] << ")";
    }
    size_t count = 1;
    out_index += IntToSize(index) * count;
    output_addr[out_index] = values_addr[0];
  } else {
    for (size_t i = 0; i < indices_shape_[0]; ++i) {
      if (i >= values_length && values_size_ != 1) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the index of 'values' out of bounds.";
      }
      size_t out_index = 0;
      for (size_t j = 0; j < rank; j++) {
        if (i * rank + j >= indices_length) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the index of 'indices' out of bounds.";
        }
        int index = indices_addr[i * rank + j];
        if (index >= output_shape_addr[j] || index < 0) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the " << i << "th value in " << j
                                   << "th dimension index: " << index << " of 'output' out of bounds: [0, "
                                   << output_shape_addr[j] << ")";
        }
        size_t count = 1;
        for (size_t k = j + 1; k < rank; k++) {
          count *= output_shape_addr[k];
        }
        out_index += IntToSize(index) * count;
      }
      if (values_size_ == 1) {
        output_addr[out_index] = values_addr[0];
      } else {
        output_addr[out_index] = values_addr[i];
      }
    }
  }
  return true;
}
const std::vector<std::pair<KernelAttr, SparseToDenseV2CpuKernelMod::KernelRunFunc>>
  &SparseToDenseV2CpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SparseToDenseV2CpuKernelMod::KernelRunFunc>> func_list = {
    {ADD_KERNEL(Int32, Int32, Bool, Bool, Bool), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, bool>},
    {ADD_KERNEL(Int32, Int32, Int8, Int8, Int8), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, int8_t>},
    {ADD_KERNEL(Int32, Int32, Int16, Int16, Int16), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, int16_t>},
    {ADD_KERNEL(Int32, Int32, Int32, Int32, Int32), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {ADD_KERNEL(Int32, Int32, Int64, Int64, Int64), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {ADD_KERNEL(Int32, Int32, UInt8, UInt8, UInt8), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, uint8_t>},
    {ADD_KERNEL(Int32, Int32, UInt16, UInt16, UInt16), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, uint16_t>},
    {ADD_KERNEL(Int32, Int32, Float16, Float16, Float16), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, float16>},
    {ADD_KERNEL(Int32, Int32, Float32, Float32, Float32), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, float>},
    {ADD_KERNEL(Int32, Int32, Float64, Float64, Float64), &SparseToDenseV2CpuKernelMod::LaunchKernel<int32_t, double>},
    {ADD_KERNEL(Int64, Int64, Bool, Bool, Bool), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, bool>},
    {ADD_KERNEL(Int64, Int64, Int8, Int8, Int8), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, int8_t>},
    {ADD_KERNEL(Int64, Int64, Int16, Int16, Int16), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, int16_t>},
    {ADD_KERNEL(Int64, Int64, Int32, Int32, Int32), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {ADD_KERNEL(Int64, Int64, Int64, Int64, Int64), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {ADD_KERNEL(Int64, Int64, UInt8, UInt8, UInt8), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, uint8_t>},
    {ADD_KERNEL(Int64, Int64, UInt16, UInt16, UInt16), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, uint16_t>},
    {ADD_KERNEL(Int64, Int64, Float16, Float16, Float16), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, float16>},
    {ADD_KERNEL(Int64, Int64, Float32, Float32, Float32), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, float>},
    {ADD_KERNEL(Int64, Int64, Float64, Float64, Float64), &SparseToDenseV2CpuKernelMod::LaunchKernel<int64_t, double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseToDenseV2, SparseToDenseV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
