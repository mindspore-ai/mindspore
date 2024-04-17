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
#ifndef AICPU_KERNELS_NORMALIZED_FFTNBASE_H_
#define AICPU_KERNELS_NORMALIZED_FFTNBASE_H_

#include <vector>
#include <securec.h>
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
const uint32_t kIndex0 = 0;
const uint32_t kFftSIndex = 1;
const uint32_t kFftDimIndex = 2;
const uint32_t kFftNormIndex = 3;
class FFTNBaseCpuKernel : public CpuKernel {
 public:
  ~FFTNBaseCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T_in, typename T_mid, typename T_out>
  uint32_t FFTNBaseCompute(CpuKernelContext &ctx);

  void FFTNGetInputs(CpuKernelContext &ctx);

  std::string op_name_;
  std::size_t dim_index_ = kFftDimIndex;
  std::size_t norm_index_ = kFftNormIndex;
  bool s_is_null_{false};
  bool dim_is_null_{false};
  std::vector<int64_t> s_;
  std::vector<int64_t> dim_;
};
}  // namespace aicpu
#endif  //  AICPU_FFTNBASE_H