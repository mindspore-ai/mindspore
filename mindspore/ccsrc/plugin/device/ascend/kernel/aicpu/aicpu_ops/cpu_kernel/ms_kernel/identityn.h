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

#ifndef AICPU_KERNELS_NORMALIZED_IDENTITY_N_H_
#define AICPU_KERNELS_NORMALIZED_IDENTITY_N_H_

#include "cpu_ops_kernel.h"

namespace aicpu {
class IdentityNCpuKernel : public CpuKernel {
 public:
  IdentityNCpuKernel() = default;
  ~IdentityNCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t IdentityNParamCheck(CpuKernelContext &ctx);
  const std::vector<DataType> support_data_type = {DT_FLOAT, DT_FLOAT16, DT_INT8,   DT_INT16,  DT_UINT16, DT_UINT8,
                                                   DT_INT32, DT_INT64,   DT_UINT32, DT_UINT64, DT_BOOL,   DT_DOUBLE};
};
}  // namespace aicpu
#endif
