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

#include "kernel/gpu/cuda/arrays/ones_gpu_kernel.h"
#include <cstdint>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/fill_v2_impl.cuh"
#include "kernel/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

int OnesGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
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
bool OnesGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *output_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  T *dev_value = nullptr;
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMalloc(reinterpret_cast<void **>(&dev_value), sizeof(T)),
                                     "Malloc slice data failed.");
  if constexpr (std::is_same<T, half>::value) {
    float16 host_value = float16(1);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(dev_value, &host_value, sizeof(float16), cudaMemcpyHostToDevice, cuda_stream_),
      "Memcpy slice data from host to device failed.");
  } else {
    T host_value = T(1);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(dev_value, &host_value, sizeof(T), cudaMemcpyHostToDevice, cuda_stream_),
      "Memcpy slice data from host to device failed.");
  }
  auto status = FillV2(outputs[kIndex0]->size() / sizeof(T), dev_value, output_ptr, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaFree(dev_value), "Free slice data failed.");
  return true;
}

// In Kernel, the type of mstype is kNumberTypeInt64;
#define ONES_GPU_REG(MS_T, MS_S, T)                            \
  KernelAttr()                                                 \
    .AddInputAttr(kObjectTypeTuple, MS_T)                      \
    .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_S),                                      \
    &OnesGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, OnesGpuKernelMod::OnesLaunchFunc>> OnesGpuKernelMod::func_list_ = {
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeFloat16, half)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeFloat32, float)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeFloat64, double)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeInt8, int8_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeInt16, int16_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeInt32, int32_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeUInt8, uint8_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeUInt16, uint16_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeUInt32, uint32_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeUInt64, uint64_t)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeBool, bool)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeComplex64, Complex<float>)},
  {ONES_GPU_REG(kNumberTypeInt64, kNumberTypeComplex128, Complex<double>)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeFloat16, half)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeFloat32, float)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeFloat64, double)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeInt8, int8_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeInt16, int16_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int64_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeUInt8, uint8_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeUInt16, uint16_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeUInt32, uint32_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeUInt64, uint64_t)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeBool, bool)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeComplex64, Complex<float>)},
  {ONES_GPU_REG(kNumberTypeInt32, kNumberTypeComplex128, Complex<double>)}};

std::vector<KernelAttr> OnesGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, OnesGpuKernelMod::OnesLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Ones, OnesGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
