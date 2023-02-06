/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_MAX_UNPOOL3D_GRAD_H_
#define AICPU_KERNELS_NORMALIZED_MAX_UNPOOL3D_GRAD_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"
namespace aicpu {
class MaxUnpool3DGradCpuKernel : public CpuKernel {
 public:
  MaxUnpool3DGradCpuKernel() = default;
  ~MaxUnpool3DGradCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static uint32_t MaxUnpool3DGradCheck(CpuKernelContext &ctx);

  template <typename T>
  static uint32_t MaxUnpool3DGrad_COMPUTE_CASE(CpuKernelContext &ctx, DataType indices_type);

  template <typename T, typename S>
  static uint32_t MaxUnpool3DGradCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_MAX_UNPOOL3D_GRAD_H_
