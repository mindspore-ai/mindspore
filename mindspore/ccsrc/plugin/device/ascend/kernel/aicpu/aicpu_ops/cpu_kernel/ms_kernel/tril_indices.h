/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All right reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_TRIL_INDICES_H_
#define AICPU_KERNELS_NORMALIZED_TRIL_INDICES_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
class TrilIndicesCpuKernel : public CpuKernel {
 public:
  TrilIndicesCpuKernel() = default;
  ~TrilIndicesCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  int32_t offset = 0;
};
}  // namespace aicpu
#endif
