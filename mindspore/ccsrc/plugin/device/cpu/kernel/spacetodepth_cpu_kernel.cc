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

#include "plugin/device/cpu/kernel/spacetodepth_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kSpaceToDepthInputsNum = 1;
constexpr size_t kSpaceToDepthOutputsNum = 1;
constexpr size_t kSpaceToDepthInputShapeSize = 4;
constexpr size_t kSpaceToDepthMinBlockSize = 2;
}  // namespace
bool SpaceToDepthCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto node_pointer = std::dynamic_pointer_cast<ops::SpaceToDepth>(base_operator);
  block_size_ = LongToSize(node_pointer->get_block_size());
  if (block_size_ < kSpaceToDepthMinBlockSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'block_size' must be greater than or equal to "
                      << kSpaceToDepthMinBlockSize << ", but got " << block_size_;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}
int SpaceToDepthCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  size_t input_num = inputs.size();
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_num;
  }

  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  if (input_shape_.size() != kSpaceToDepthInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input tensor must be 4-D, but got "
                      << input_shape_.size() << "-D";
  }
  output_shape_ = outputs[0]->GetShapeVector();
  return KRET_OK;
}
template <typename T>
bool SpaceToDepthCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSpaceToDepthInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSpaceToDepthOutputsNum, kernel_name_);
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t size = inputs[0]->size / sizeof(T);

  auto input_shape = input_shape_;
  auto output_shape = output_shape_;
  int64_t block_size = SizeToLong(block_size_);
  size_t input_dimension = input_shape.size();
  int64_t input_strides[3] = {1, 1, 1};

  for (size_t i = input_dimension - 1; i >= 1; --i) {
    for (size_t j = 0; j < i; ++j) {
      input_strides[j] *= input_shape[i];
    }
  }

  auto task = [&, input_addr, output_addr](size_t start, size_t end) {
    std::vector<int64_t> input_pos_array(input_dimension, 0);
    for (size_t i = start; i < end; ++i) {
      int64_t tmp_pos = SizeToLong(i);
      for (size_t j = 0; j < input_dimension - 1; ++j) {
        input_pos_array[j] = tmp_pos / input_strides[j];
        tmp_pos %= input_strides[j];
      }
      input_pos_array.back() = tmp_pos;
      int64_t output_pos = input_pos_array[0];
      output_pos = (output_pos * output_shape[1]) +
                   (input_pos_array[1] + (block_size * (input_pos_array[2] % SizeToLong(block_size)) +
                                          input_pos_array[3] % SizeToLong(block_size)) *
                                           input_shape[1]);
      output_pos = (output_pos * output_shape[2]) + (input_pos_array[2] / SizeToLong(block_size));
      output_pos = (output_pos * output_shape[3]) + (input_pos_array[3] / SizeToLong(block_size));
      output_addr[output_pos] = input_addr[i];
    }
  };

  ParallelLaunchAutoSearch(task, size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, SpaceToDepthCpuKernelMod::SpaceToDepthFunc>> SpaceToDepthCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SpaceToDepthCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SpaceToDepthCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &SpaceToDepthCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &SpaceToDepthCpuKernelMod::LaunchKernel<complex128>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint64_t>}};

std::vector<KernelAttr> SpaceToDepthCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpaceToDepthFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SpaceToDepth, SpaceToDepthCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
