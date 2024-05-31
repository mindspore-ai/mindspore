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
#ifndef AICPU_KERNELS_NORMALIZED_FFTBASE_H_
#define AICPU_KERNELS_NORMALIZED_FFTBASE_H_

#include <vector>
#include <securec.h>
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
const uint32_t kIndex0 = 0;
const uint32_t kFftNIndex = 1;
const uint32_t kFftDimIndex = 2;
const uint32_t kFftNormIndex = 3;
class FFTBaseCpuKernel : public CpuKernel {
 public:
  ~FFTBaseCpuKernel() = default;

  DataType input_type_;
  DataType output_type_;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

  template <typename T_in, typename T_out>
  static uint32_t FFTBaseCompute(CpuKernelContext &ctx);

  template <typename T_in, typename T_out>
  static uint32_t FFTBaseComputeC2C(CpuKernelContext &ctx);

  template <typename T_in, typename T_out>
  static uint32_t FFTBaseComputeR2C(CpuKernelContext &ctx);

  template <typename T_in, typename T_out>
  static uint32_t FFTBaseComputeC2R(CpuKernelContext &ctx);

 private:
  uint32_t ParseKernelParam(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  //  AICPU_FFTBASE_H
