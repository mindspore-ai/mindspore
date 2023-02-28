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

#include "plugin/device/cpu/kernel/masked_scatter_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <functional>
#include "mindspore/core/ops/masked_scatter.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaskedScatterInputsNum = 3;
constexpr size_t kMaskedScatterOutputsNum = 1;
}  // namespace

bool MaskedScatterCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "MaskedScatter does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaskedScatterCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  mask_shape_ = inputs.at(kIndex1)->GetShapeVector();
  updates_shape_ = inputs.at(kIndex2)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  x_numElements_ = std::accumulate(x_shape_.begin(), x_shape_.end(), 1, std::multiplies<size_t>());
  updates_numElements_ = std::accumulate(updates_shape_.begin(), updates_shape_.end(), 1, std::multiplies<size_t>());
  need_broadcast_ = (x_shape_ == mask_shape_) ? false : true;
  size_t mask_dims = mask_shape_.size();
  std::vector<int64_t> x_shape_reverse = x_shape_;
  std::vector<int64_t> mask_shape_reverse = mask_shape_;
  std::reverse(x_shape_reverse.begin(), x_shape_reverse.end());
  std::reverse(mask_shape_reverse.begin(), mask_shape_reverse.end());
  for (size_t i = 0; i < mask_dims; i++) {
    if (mask_shape_reverse[i] != x_shape_reverse[i] && mask_shape_reverse[i] != 1) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the shape of 'mask': " << ShapeVectorToStr(mask_shape_)
                               << " can not be broadcast to the shape of 'x': " << ShapeVectorToStr(x_shape_) << ".";
    }
  }
  return ret;
}

template <typename T>
bool MaskedScatterCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedScatterInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedScatterOutputsNum, kernel_name_);

  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto mask = reinterpret_cast<bool *>(inputs[1]->addr);
  auto updates = reinterpret_cast<T *>(inputs[2]->addr);
  auto y = reinterpret_cast<T *>(outputs[0]->addr);
  uint64_t j = 0;
  if (!need_broadcast_) {
    for (uint64_t i = 0; i < x_numElements_; i++) {
      if (mask[i]) {
        if (j >= updates_numElements_) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                                   << "', number of elements of updates < number of ones in mask.";
        }
        y[i] = updates[j], j += 1;
      } else {
        y[i] = x[i];
      }
    }
  } else {
    BroadcastIterator iter(x_shape_, mask_shape_, output_shape_);
    iter.SetPos(0);
    for (uint64_t i = 0; i < x_numElements_; i++, iter.GenNextPos()) {
      if (mask[iter.GetInputPosB()]) {
        if (j >= updates_numElements_) {
          MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                                   << "', number of elements of updates < number of ones in mask.";
        }
        y[iter.GetInputPosA()] = updates[j], j += 1;
      } else {
        y[i] = x[i];
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, MaskedScatterCpuKernelMod::MaskedScatterFunc>> MaskedScatterCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
    &MaskedScatterCpuKernelMod::LaunchKernel<float16>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    &MaskedScatterCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat64),
    &MaskedScatterCpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeUInt8)
      .AddOutputAttr(kNumberTypeUInt8),
    &MaskedScatterCpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt8)
      .AddOutputAttr(kNumberTypeInt8),
    &MaskedScatterCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt16)
      .AddOutputAttr(kNumberTypeInt16),
    &MaskedScatterCpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    &MaskedScatterCpuKernelMod::LaunchKernel<int32_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    &MaskedScatterCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> MaskedScatterCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaskedScatterFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaskedScatter, MaskedScatterCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
