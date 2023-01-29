/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/concat_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/concat.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kConcatOutputsNum = 1;
}  // namespace

bool ConcatCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Concat does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Concat>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  ori_axis_ = kernel_ptr->get_axis();
  return true;
}

int ConcatCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  inputs_shape_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs_shape_.push_back(inputs[i]->GetShapeVector());
  }
  axis_ = ori_axis_;
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(inputs_shape_[0].size());
  }
  return KRET_OK;
}

template <typename T>
bool ConcatCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  const size_t input_num = inputs.size();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConcatOutputsNum, kernel_name_);

  std::vector<ShapeVector> input_flat_shape_list;
  input_flat_shape_list.reserve(input_num);
  for (size_t i = 0; i < input_num; i++) {
    auto input_shape_i = inputs_shape_[i];
    auto flat_shape = CPUKernelUtils::FlatShapeByAxis(input_shape_i, axis_);
    (void)input_flat_shape_list.emplace_back(flat_shape);
  }

  size_t output_dim_1 = 0;
  for (size_t j = 0; j < input_num; ++j) {
    output_dim_1 += LongToSize(input_flat_shape_list[j][1]);
  }
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  std::vector<T *> input_addr_list;
  for (size_t j = 0; j < input_num; ++j) {
    auto *tmp_addr = reinterpret_cast<T *>(inputs[j]->addr);
    (void)input_addr_list.emplace_back(tmp_addr);
  }
  if (input_flat_shape_list.size() == 0 || input_flat_shape_list[0].size() == 0) {
    return true;
  }
  // each input's row of shape after flat are same
  auto before_axis = LongToSize(input_flat_shape_list[0][0]);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      auto output_ptr = output_addr + i * output_dim_1;
      for (size_t j = 0; j < input_num; ++j) {
        if (input_flat_shape_list[j][1] == 0) {
          continue;
        }
        auto copy_num = LongToSize(input_flat_shape_list[j][1]);
        auto copy_size = copy_num * sizeof(T);
        auto offset = copy_num * i;
        auto ret = memcpy_s(output_ptr, copy_size, input_addr_list[j] + offset, copy_size);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed. Error no: " << ret;
        }
        output_ptr += copy_num;
      }
    }
  };
  ParallelLaunchAutoSearch(task, before_axis, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, ConcatCpuKernelMod::ConcatFunc>> ConcatCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ConcatCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ConcatCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ConcatCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &ConcatCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &ConcatCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &ConcatCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ConcatCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &ConcatCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &ConcatCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &ConcatCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &ConcatCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &ConcatCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &ConcatCpuKernelMod::LaunchKernel<complex128>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &ConcatCpuKernelMod::LaunchKernel<bool>}};

std::vector<KernelAttr> ConcatCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ConcatFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Concat, ConcatCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
