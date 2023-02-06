
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

#ifndef AICPU_KERNELS_NORMALIZED_MAX_POOL3D_WINTH_ARGMAX_H_
#define AICPU_KERNELS_NORMALIZED_MAX_POOL3D_WINTH_ARGMAX_H_

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class MaxPool3DWithArgmaxCpuKernel : public CpuKernel {
 public:
  MaxPool3DWithArgmaxCpuKernel() = default;
  ~MaxPool3DWithArgmaxCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t MaxPool3DWithArgmaxParamCheck(CpuKernelContext &ctx);

  template <typename T, typename S>
  uint32_t MaxPool3DWithArgmaxCompute(CpuKernelContext &ctx);

  template <typename T, typename S>
  void MaxPool3DWithArgmaxSingleCompute(T *input, T *output_y, S *output_argmax, int64_t iD, int64_t iH, int64_t iW,
                                        int64_t oD, int64_t oH, int64_t oW, int64_t kD, int64_t kH, int64_t kW,
                                        int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW,
                                        int64_t dD, int64_t dH, int64_t dW);
};
}  // namespace aicpu
#endif