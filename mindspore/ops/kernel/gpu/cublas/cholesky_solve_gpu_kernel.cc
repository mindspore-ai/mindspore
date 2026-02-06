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

#include "kernel/gpu/cublas/cholesky_solve_gpu_kernel.h"
#include "mindspore/ops/infer/cholesky_solve.h"

namespace mindspore {
namespace kernel {
using CSGKM = CholeskySolveGpuKernelMod;
std::vector<std::pair<KernelAttr, CSGKM::CholeskySolveFunc>> CSGKM::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &CholeskySolveGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &CholeskySolveGpuKernelMod::LaunchKernel<double>},
};
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CholeskySolve, CholeskySolveGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
