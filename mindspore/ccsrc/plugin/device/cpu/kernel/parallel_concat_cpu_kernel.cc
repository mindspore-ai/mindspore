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

#include "plugin/device/cpu/kernel/parallel_concat_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int axis = 0;
constexpr size_t kParallelConcatOutputsNum = 1;
}  // namespace

bool ParallelConcatCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the data type of input must be float or double, but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ParallelConcatCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  int64_t output_elements = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements == 0) {
    is_null_input_ = true;
  }
  input_num_ = inputs.size();
  auto x_shape = inputs[0]->GetShapeVector();
  for (size_t i = 0; i < input_num_; i++) {
    if (x_shape != inputs[i]->GetShapeVector()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of all tensors must be the same, but got tensor0.shape "
                    << x_shape << " and tensor" << i << ".shape " << inputs[i]->GetShapeVector();
    }
  }
  input_flat_shape_list_.reserve(input_num_);
  for (size_t i = 0; i < input_num_; i++) {
    auto input_shape_i = inputs[i]->GetShapeVector();
    auto flat_shape = CPUKernelUtils::FlatShapeByAxis(input_shape_i, axis);
    (void)input_flat_shape_list_.emplace_back(flat_shape);
  }
  return KRET_OK;
}

template <typename T>
bool ParallelConcatCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  std::vector<T *> input_addr_list;
  for (size_t j = 0; j < input_num_; ++j) {
    auto *tmp_addr = static_cast<T *>(inputs[j]->addr);
    (void)input_addr_list.emplace_back(tmp_addr);
  }
  auto *output_addr = static_cast<T *>(outputs[0]->addr);

  size_t output_dim_1 = 0;
  for (size_t j = 0; j < input_num_; ++j) {
    output_dim_1 += LongToSize(input_flat_shape_list_[j][1]);
  }

  // each input's row of shape after flat are same
  auto before_axis = LongToSize(input_flat_shape_list_[0][0]);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      auto output_ptr = output_addr + i * output_dim_1;
      for (size_t j = 0; j < input_num_; ++j) {
        if (input_flat_shape_list_[j][1] == 0) {
          continue;
        }
        auto copy_num = LongToSize(input_flat_shape_list_[j][1]);
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

std::vector<std::pair<KernelAttr, ParallelConcatCpuKernelMod::PCFunc>> ParallelConcatCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &ParallelConcatCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &ParallelConcatCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &ParallelConcatCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &ParallelConcatCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &ParallelConcatCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &ParallelConcatCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &ParallelConcatCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ParallelConcatCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ParallelConcatCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ParallelConcatCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ParallelConcatCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &ParallelConcatCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &ParallelConcatCpuKernelMod::LaunchKernel<complex128>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &ParallelConcatCpuKernelMod::LaunchKernel<bool>}};

std::vector<KernelAttr> ParallelConcatCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, PCFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ParallelConcat, ParallelConcatCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
