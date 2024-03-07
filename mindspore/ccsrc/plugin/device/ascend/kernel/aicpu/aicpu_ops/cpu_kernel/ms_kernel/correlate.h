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
#ifndef AICPU_KERNELS_NORMALIZED_CORRELATE_H_
#define AICPU_KERNELS_NORMALIZED_CORRELATE_H_

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class CorrelateCpuKernel : public CpuKernel {
 public:
  CorrelateCpuKernel() = default;
  ~CorrelateCpuKernel() override = default;
  template <typename T>
  void CorrelatePad(T *source_array, T *padded_array, int64_t padded_array_size, int64_t long_size, int64_t short_size,
                    std::string mode);

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T_in, typename T_out>
  uint32_t CorrelateCompute(CpuKernelContext &ctx);
  template <typename T>
  uint32_t CorrelateComputeComplex(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
