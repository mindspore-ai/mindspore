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
#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_

#include <vector>
#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalMaxPoolWithFixedKsize : public CpuKernel {
 public:
  FractionalMaxPoolWithFixedKsize() = default;
  ~FractionalMaxPoolWithFixedKsize() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T, typename SType>
  uint32_t FractionalMaxPoolWithFixedKsizeCompute(Tensor *x, Tensor *random_samples, const int input_n,
                                                  const int input_c, const int input_h, const int input_w,
                                                  const int output_h, const int output_w, const int pool_h,
                                                  const int pool_w, CpuKernelContext &ctx);
  template <typename T, typename SType>
  uint32_t ComputeSingleBatch(T *x_addr, SType *random_samples_addr, T *y_addr, int64_t *argmax_addr, const int input_c,
                              const int input_h, const int input_w, const int output_h, const int output_w,
                              const int pool_h, const int pool_w);
  template <typename SType>
  std::vector<int> FractionalMaxPoolWithFixedKsizeGenerateIntervals(SType sample, const int input_size,
                                                                    const int output_size, const int pool_size);
};
}  // namespace aicpu
#endif
