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

#include "plugin/device/cpu/kernel/depthtospace_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kDepthToSpaceInputsNum = 1;
constexpr size_t kDepthToSpaceOutputsNum = 1;
constexpr size_t kDepthToSpaceInputDimension = 4;
}  // namespace

bool DepthToSpaceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::DepthToSpace>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  block_size_ = LongToSize(kernel_ptr->get_block_size());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int DepthToSpaceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  return ret;
}

template <typename T>
bool DepthToSpaceCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDepthToSpaceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDepthToSpaceOutputsNum, kernel_name_);
  auto input_rank = input_shape_.size();
  if (input_rank != kDepthToSpaceInputDimension) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input should have a rank of 4, but got input of rank "
                      << input_rank;
  }
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t size = inputs[0]->size / sizeof(T);
  auto input_shape = input_shape_;
  auto output_shape = output_shape_;
  size_t block_size = block_size_;
  size_t input_dimension = input_shape.size();
  size_t output_strides[3] = {1, 1, 1};

  for (size_t i = input_dimension - 1; i >= 1; --i) {
    for (size_t j = 0; j < i; ++j) {
      output_strides[j] *= static_cast<size_t>(output_shape[i]);
    }
  }

  auto task = [&, input_addr, output_addr](size_t start, size_t end) {
    std::vector<size_t> output_pos_array(input_dimension, 0);
    for (size_t i = start; i < end; ++i) {
      size_t tmp_pos = i;
      for (size_t j = 0; j < input_dimension - 1; ++j) {
        output_pos_array[j] = tmp_pos / output_strides[j];
        tmp_pos %= output_strides[j];
      }
      output_pos_array.back() = tmp_pos;
      size_t input_pos = output_pos_array[0];
      input_pos =
        (input_pos * static_cast<size_t>(input_shape[1])) +
        (output_pos_array[1] + (block_size * (output_pos_array[2] % block_size) + output_pos_array[3] % block_size) *
                                 static_cast<size_t>(output_shape[1]));
      input_pos = (input_pos * static_cast<size_t>(input_shape[2])) + (output_pos_array[2] / block_size);
      input_pos = (input_pos * static_cast<size_t>(input_shape[3])) + (output_pos_array[3] / block_size);
      output_addr[i] = input_addr[input_pos];
    }
  };

  ParallelLaunchAutoSearch(task, size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, DepthToSpaceCpuKernelMod::DepthToSpaceFunc>> DepthToSpaceCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &DepthToSpaceCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &DepthToSpaceCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &DepthToSpaceCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &DepthToSpaceCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &DepthToSpaceCpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &DepthToSpaceCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &DepthToSpaceCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &DepthToSpaceCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &DepthToSpaceCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &DepthToSpaceCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &DepthToSpaceCpuKernelMod::LaunchKernel<complex128>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &DepthToSpaceCpuKernelMod::LaunchKernel<uint64_t>}};

std::vector<KernelAttr> DepthToSpaceCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DepthToSpaceFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DepthToSpace, DepthToSpaceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
