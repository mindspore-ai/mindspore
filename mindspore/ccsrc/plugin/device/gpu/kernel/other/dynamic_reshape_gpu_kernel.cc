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
#include "plugin/device/gpu/kernel/other/dynamic_reshape_gpu_kernel.h"
#include <iterator>
#include <algorithm>
#include <functional>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, DynamicReshapeKernelMod::LaunchFunc>> DynamicReshapeKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &DynamicReshapeKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &DynamicReshapeKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &DynamicReshapeKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &DynamicReshapeKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &DynamicReshapeKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &DynamicReshapeKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &DynamicReshapeKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &DynamicReshapeKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &DynamicReshapeKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &DynamicReshapeKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &DynamicReshapeKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &DynamicReshapeKernelMod::LaunchKernel<int64_t>}};

template <typename S>
bool DynamicReshapeKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto data_addr = GetDeviceAddress<unsigned char>(inputs, 0);
  auto shape_addr = GetDeviceAddress<S>(inputs, 1);
  auto output_addr = GetDeviceAddress<unsigned char>(outputs, 0);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output_addr, data_addr, input_size_list_[0], cudaMemcpyDeviceToDevice, cuda_stream),
    "DynamicReshape cpy data failed");
  std::vector<S> real_output_shape_ = std::vector<S>(input_size_list_[1] / sizeof(S), 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&real_output_shape_[0], shape_addr, input_size_list_[1], cudaMemcpyDeviceToHost, cuda_stream),
    "DynamicReshape cpy real output shape value failed");
  std::transform(real_output_shape_.begin(), real_output_shape_.end(), std::back_inserter(output_shape_),
                 [](const S &value) { return static_cast<int64_t>(value); });
  return true;
}

std::vector<KernelAttr> DynamicReshapeKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, DynamicReshapeKernelMod::LaunchFunc> &pair) { return pair.first; });
  return support_list;
}
}  // namespace kernel
}  // namespace mindspore
