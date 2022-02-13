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

#include "plugin/device/gpu/kernel/math/broadcast_complex_gpu_kernel.h"

namespace mindspore {
namespace kernel {

#define MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(OPNAME, T0_MS_DTYPE, T1_MS_DTYPE, T0_DTYPE, T1_DTYPE)                      \
  MS_REG_GPU_KERNEL_THREE(OPNAME,                                                                                      \
                          KernelAttr().AddInputAttr(T0_MS_DTYPE).AddInputAttr(T0_MS_DTYPE).AddOutputAttr(T0_MS_DTYPE), \
                          BroadcastComplexOpGpuKernelMod, T0_DTYPE, T0_DTYPE, T0_DTYPE)                                \
  MS_REG_GPU_KERNEL_THREE(OPNAME,                                                                                      \
                          KernelAttr().AddInputAttr(T0_MS_DTYPE).AddInputAttr(T1_MS_DTYPE).AddOutputAttr(T0_MS_DTYPE), \
                          BroadcastComplexOpGpuKernelMod, T0_DTYPE, T1_DTYPE, T0_DTYPE)                                \
  MS_REG_GPU_KERNEL_THREE(OPNAME,                                                                                      \
                          KernelAttr().AddInputAttr(T1_MS_DTYPE).AddInputAttr(T0_MS_DTYPE).AddOutputAttr(T0_MS_DTYPE), \
                          BroadcastComplexOpGpuKernelMod, T1_DTYPE, T0_DTYPE, T0_DTYPE)

template <typename T>
using Complex = mindspore::utils::Complex<T>;
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Add, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Add, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Sub, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Sub, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Mul, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Mul, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Div, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(Div, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(RealDiv, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float);
MS_REG_BROADCAST_COMPLEX_GPU_KERNEL(RealDiv, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double);
MS_REG_GPU_KERNEL_THREE(
  Complex,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
  BroadcastComplexOpGpuKernelMod, float, float, Complex<float>)
MS_REG_GPU_KERNEL_THREE(
  Complex,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
  BroadcastComplexOpGpuKernelMod, double, double, Complex<double>)
}  // namespace kernel
}  // namespace mindspore
