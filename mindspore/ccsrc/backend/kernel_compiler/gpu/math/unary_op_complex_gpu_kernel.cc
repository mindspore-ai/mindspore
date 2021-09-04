/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "utils/complex.h"
#include "backend/kernel_compiler/gpu/math/unary_op_complex_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Real, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpComplexGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(Real, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpComplexGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(Imag, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpComplexGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(Imag, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpComplexGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(Conj, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                      UnaryOpComplexGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(Conj, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                      UnaryOpComplexGpuKernel, double)
}  // namespace kernel
}  // namespace mindspore
