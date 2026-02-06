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

#include "kernel/gpu/cuda/arrays/eye_gpu_kernel.h"

#include <algorithm>
#include <memory>

#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/eye_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
namespace {
constexpr size_t kEyeInputsNum = 3;
constexpr size_t kEyeOutputsNum = 1;
}  // namespace
bool EyeGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int EyeGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }
  int ret = 0;
  if ((ret = NativeGpuKernelMod::Resize(inputs, outputs)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  auto out_shape = outputs.at(kIndex0)->GetDeviceShapeVector();
  num_n_ = out_shape[kIndex0];
  num_m_ = out_shape[kIndex1];
  return 0;
}

template <typename T>
bool EyeGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  T *ouput_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  auto status =
    CudaEye(outputs[kIndex0]->size(), num_n_, num_m_, ouput_ptr, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

// In Kernel, the type of mstype is kNumberTypeInt64;
#define EYE_GPU_REG(MS_T, MS_S, T)                                                                       \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddInputAttr(kNumberTypeInt64).AddOutputAttr(MS_S), \
    &EyeGpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, EyeGpuKernelMod::KernelRunFunc>> &EyeGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, EyeGpuKernelMod::KernelRunFunc>> func_list = {
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeFloat16, half)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeFloat16, half)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeFloat32, float)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeFloat32, float)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeFloat64, double)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeFloat64, double)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeInt8, int8_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeInt8, int8_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeInt16, int16_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeInt16, int16_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeInt32, int32_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int64_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeUInt8, uint8_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeUInt8, uint8_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeUInt16, uint16_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeUInt16, uint16_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeUInt32, uint32_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeUInt32, uint32_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeUInt64, uint64_t)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeUInt64, uint64_t)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeComplex64, Complex<float>)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeComplex64, Complex<float>)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeComplex128, Complex<double>)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeComplex128, Complex<double>)},
    {EYE_GPU_REG(kNumberTypeInt32, kNumberTypeBool, bool)},
    {EYE_GPU_REG(kNumberTypeInt64, kNumberTypeBool, bool)}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Eye, EyeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
