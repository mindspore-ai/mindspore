/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved
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

#ifndef AICPU_KERNELS_NORMALIZED_LUUNPACK_H_
#define AICPU_KERNELS_NORMALIZED_LUUNPACK_H_

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"
namespace aicpu {
class LuUnpackCpuKernel : public CpuKernel {
 public:
  LuUnpackCpuKernel() = default;
  ~LuUnpackCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T_data, typename T_pivots>
  static uint32_t LuUnpack(CpuKernelContext &ctx, T_pivots *Lu_pivots_working_ptr, int64_t matrix_index, T_data *P_eye);
  template <typename T_data, typename T_pivots>
  static uint32_t LuUnpackCompute(CpuKernelContext &ctx);
  template <typename T_pivots>
  static uint32_t DataAndTypeCheck(CpuKernelContext &ctx);
  std::map<int, std::map<int, std::function<uint32_t(CpuKernelContext &)>>> calls_;
  void SetMap();
};
}  // namespace aicpu
#endif
