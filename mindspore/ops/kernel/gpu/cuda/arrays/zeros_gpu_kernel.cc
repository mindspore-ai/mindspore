/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/zeros_gpu_kernel.h"
#include <cstdint>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

int ZerosGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return KRET_RESIZE_FAILED;
  }
  kernel_func_ = func_list_[index].second;
  return KRET_OK;
}

template <typename T>
bool ZerosGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  void *output_device_address = outputs[kIndex0]->device_ptr();
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    // have to use a float literal instead of an int literal because of ambiguous half() overload.
    cudaMemsetAsync(output_device_address, 0, outputs[kIndex0]->size(), reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemset failed");
  return true;
}

// In Kernel, the type of mstype is kNumberTypeInt64;
#define ZEROS_GPU_REG(MS_T, MS_S, T)                           \
  KernelAttr()                                                 \
    .AddInputAttr(kObjectTypeTuple, MS_T)                      \
    .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_S),                                      \
    &ZerosGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ZerosGpuKernelMod::ZerosLaunchFunc>> ZerosGpuKernelMod::func_list_ = {
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeFloat16, half)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeFloat32, float)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeFloat64, double)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeInt8, int8_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeInt16, int16_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeInt32, int32_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeUInt8, uint8_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeUInt16, uint16_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeUInt32, uint32_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeUInt64, uint64_t)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeBool, bool)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeComplex64, Complex<float>)},
  {ZEROS_GPU_REG(kNumberTypeInt64, kNumberTypeComplex128, Complex<double>)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeFloat16, half)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeFloat32, float)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeFloat64, double)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeInt8, int8_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeInt16, int16_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int64_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeUInt8, uint8_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeUInt16, uint16_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeUInt32, uint32_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeUInt64, uint64_t)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeBool, bool)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeComplex64, Complex<float>)},
  {ZEROS_GPU_REG(kNumberTypeInt32, kNumberTypeComplex128, Complex<double>)}};

std::vector<KernelAttr> ZerosGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ZerosGpuKernelMod::ZerosLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Zeros, ZerosGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
