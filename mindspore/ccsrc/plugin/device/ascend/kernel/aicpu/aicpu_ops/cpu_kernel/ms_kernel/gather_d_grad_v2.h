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

#ifndef AICPU_KERNELS_NORMALIZED_GATHER_D_GRAD_V2_H_
#define AICPU_KERNELS_NORMALIZED_GATHER_D_GRAD_V2_H_

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class GatherDGradV2Kernel : public CpuKernel {
 public:
  GatherDGradV2Kernel() = default;
  ~GatherDGradV2Kernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParseKernelParam(CpuKernelContext &ctx);

  template <typename T, typename S>
  uint32_t GatherDGradV2Task(CpuKernelContext &ctx);

  int64_t dim_{0};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> index_shape_;
  std::vector<int64_t> grad_shape_;
  std::vector<int64_t> output_shape_;

  DataType index_type_{DT_UNDEFINED};
  DataType grad_type_{DT_UNDEFINED};
};
}  // namespace aicpu
#endif