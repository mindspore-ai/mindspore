/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sparse_to_dense_cpu_kernal.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndicesShapeSize = 2;
constexpr size_t kSparseToDenseInputsNum = 3;
constexpr size_t kSparseToDenseOutputsNum = 1;
}  // namespace
bool SparseToDenseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SparseToDense does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseToDenseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto indices_shape = inputs.at(kIndex0)->GetShapeVector();
  if (indices_shape.size() != kIndicesShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'indices' must be a " << kIndicesShapeSize
                      << "-D Tensor, but got " << indices_shape.size() << "-D";
  }
  auto values_shape = inputs.at(kIndex1)->GetShapeVector();
  if (values_shape.size() != 1 || values_shape[0] != indices_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'values' must be a 1-D Tensor and the first dimension length "
                         "must be equal to the first dimension length of 'indices', but got 'values' shape: "
                      << Vector2Str(values_shape) << " and 'indices' shape: " << Vector2Str(indices_shape);
  }
  output_shape_ = Convert2SizeT(outputs.at(kIndex0)->GetShapeVector());
  values_size_ = LongToSize(values_shape[0]);
  return KRET_OK;
}

template <typename I, typename T>
bool SparseToDenseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseToDenseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseToDenseOutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size must be greater than 0, but got 0.";
    return true;
  }
  auto ret = memset_s(outputs[0]->addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret;
  }

  const auto *indices_addr = reinterpret_cast<I *>(inputs[0]->addr);
  const auto *values_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t indices_length = inputs[0]->size / sizeof(I);
  const size_t values_length = inputs[1]->size / sizeof(T);
  size_t rank = output_shape_.size();

  for (size_t i = 0; i < values_size_; ++i) {
    if (i >= values_length) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'values' out of bounds.";
    }
    size_t out_index = 0;
    for (size_t j = 0; j < rank; j++) {
      if (i * rank + j >= indices_length) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'indices' out of bounds.";
      }
      int index = indices_addr[i * rank + j];
      if (index >= SizeToInt(output_shape_[j]) || index < 0) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the " << i << "th value in " << j
                                 << "th dimension index: " << index << " of 'output' out of bounds: [0, "
                                 << output_shape_[j] << ")";
      }
      size_t count = 1;
      for (size_t k = j + 1; k < rank; k++) {
        count *= output_shape_[k];
      }
      out_index += IntToSize(index) * count;
    }
    output_addr[out_index] = values_addr[i];
  }
  return true;
}

std::vector<std::pair<KernelAttr, SparseToDenseCpuKernelMod::SparseToDenseFunc>> SparseToDenseCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeBool),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, bool>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt8),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, int8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt16),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, int16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, int32_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt64),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, int64_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt8),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, uint8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt16)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt16),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, uint16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt32),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, uint32_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeUInt64)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeUInt64),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, uint64_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat16),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, float16>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat64),
    &SparseToDenseCpuKernelMod::LaunchKernel<int32_t, double>}};

std::vector<KernelAttr> SparseToDenseCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseToDenseFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseToDense, SparseToDenseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
